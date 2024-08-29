import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer_prompts as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
import ipdb


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.prepare_data_per_node = False

        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"]
            * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(
            2, config["hidden_size"]
        )
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)  # 把初始化应用到所有子模块

        if config["loss_names"]["mlm"] > 0:  # Masked Language Model.
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:  # Image-Text Matching.
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:  # Masked Patch Prediction.
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["finetune_first"]
        ):

            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]

            # since the pre-trained max_text_len is 40,
            # we upsample the weight of position embedding to determined max_text_len
            if config["max_text_len"] != 40:
                state_dict["text_embeddings.position_ids"] = (
                    torch.Tensor(range(config["max_text_len"])).long().view(1, -1)
                )
                pos_emb = state_dict["text_embeddings.position_embeddings.weight"]
                pos_emb = torch.nn.functional.interpolate(
                    pos_emb.view(1, 1, 40, 768),
                    size=(config["max_text_len"], 768),
                    mode="bilinear",
                ).squeeze()
                state_dict["text_embeddings.position_embeddings.weight"] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.hatememes_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.mmimdb_classifier.apply(objectives.init_weights)

        if (
            self.hparams.config["load_path"] != ""
            and self.hparams.config["finetune_first"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        self.prompt_type = self.hparams.config["prompt_type"]
        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_prompt = self.hparams.config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1

        from timm.models.layers import trunc_normal_

        # ===================== Initializing prompts ===================== #
        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_prompt[:, 0:1, :].fill_(1)
        self.complete_prompt = nn.Parameter(complete_prompt)

        missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_text_prompt[:, 2:3, :].fill_(1)
        self.missing_text_prompt = nn.Parameter(missing_text_prompt)

        missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_img_prompt[:, 1:2, :].fill_(1)
        self.missing_img_prompt = nn.Parameter(missing_img_prompt)

        kro_prompt_A_t = torch.zeros(prompt_num, 2, 2)
        kro_prompt_A_t[:, 0, 1].fill_(1)
        kro_prompt_A_t[:, 1, 0].fill_(1)
        kro_prompt_A_i = torch.zeros(prompt_num, 2, 2)
        kro_prompt_A_i[:, 0, 0].fill_(1)
        kro_prompt_A_i[:, 1, 1].fill_(1)

        kro_prompt_B1 = torch.randn(prompt_num, int(prompt_length / 2), 3)
        kro_prompt_B2 = torch.randn(prompt_num, int(prompt_length / 2), 3)
        kro_prompt_B3 = torch.randn(prompt_num, int(prompt_length / 2), 3)
        kro_prompt_B4 = torch.randn(prompt_num, int(prompt_length / 2), 3)
        kro_prompt_C1 = torch.randn(prompt_num, 3, int(embed_dim / 2))
        kro_prompt_C2 = torch.randn(prompt_num, 3, int(embed_dim / 2))
        kro_prompt_C3 = torch.randn(prompt_num, 3, int(embed_dim / 2))
        kro_prompt_C4 = torch.randn(prompt_num, 3, int(embed_dim / 2))

        self.kro_prompt_A_t = nn.Parameter(kro_prompt_A_t)
        self.kro_prompt_A_i = nn.Parameter(kro_prompt_A_i)
        self.kro_prompt_B1 = nn.Parameter(kro_prompt_B1)
        self.kro_prompt_B2 = nn.Parameter(kro_prompt_B2)
        self.kro_prompt_B3 = nn.Parameter(kro_prompt_B3)
        self.kro_prompt_B4 = nn.Parameter(kro_prompt_B4)
        self.kro_prompt_C1 = nn.Parameter(kro_prompt_C1)
        self.kro_prompt_C2 = nn.Parameter(kro_prompt_C2)
        self.kro_prompt_C3 = nn.Parameter(kro_prompt_C3)
        self.kro_prompt_C4 = nn.Parameter(kro_prompt_C4)

        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.text_embeddings.parameters():
            param.requires_grad = False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print("the A matrices: ", 
                  state_dict['kro_prompt_A_t'], 
                  state_dict['kro_prompt_A_i'])

        self.records = {}
        self.with_delta_infer = self.hparams.config["with_delta_infer"]
        print("Now, the prompt type is: ", self.prompt_type)
        self.printed = False

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_embeddings(text_ids)

        img = batch[imgkey][0]

        ######## generate image and text embeds ########
        if image_embeds is None and image_masks is None:
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = None, None

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        def modified_kronecker_product(A, B1, B2, B3, B4, C1, C2, C3, C4):
            modified_Block1 = A[0, 0, 0] * (B1 @ C1)
            modified_Block2 = A[0, 0, 1] * (B2 @ C2)
            modified_Block3 = A[0, 1, 0] * (B3 @ C3)
            modified_Block4 = A[0, 1, 1] * (B4 @ C4)
            cat_1 = torch.cat([modified_Block1, modified_Block2], dim=2)
            cat_2 = torch.cat([modified_Block3, modified_Block4], dim=2)
            res = torch.cat([cat_1, cat_2], dim=1)
            return res

        device = next(self.parameters()).device

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        cls_feats_list = []
        text_feats_list = []
        image_feats_list = []
        raw_cls_feats_list = []

        for idx in range(len(img)):
            if self.prompt_type == "kronecker":
                if batch["missing_type"][idx] == 0:
                    prompt = (
                        modified_kronecker_product(
                            (self.kro_prompt_A_t + self.kro_prompt_A_i),
                            self.kro_prompt_B1,
                            self.kro_prompt_B2,
                            self.kro_prompt_B3,
                            self.kro_prompt_B4,
                            self.kro_prompt_C1,
                            self.kro_prompt_C2,
                            self.kro_prompt_C3,
                            self.kro_prompt_C4,
                        ).to(device)
                        if (is_train or (not is_train and self.with_delta_infer))
                        else None
                    )
                elif batch["missing_type"][idx] == 1:
                    prompt = modified_kronecker_product(
                        self.kro_prompt_A_t,
                        self.kro_prompt_B1,
                        self.kro_prompt_B2,
                        self.kro_prompt_B3,
                        self.kro_prompt_B4,
                        self.kro_prompt_C1,
                        self.kro_prompt_C2,
                        self.kro_prompt_C3,
                        self.kro_prompt_C4,
                    ).to(device)
                elif batch["missing_type"][idx] == 2:
                    prompt = modified_kronecker_product(
                        self.kro_prompt_A_i,
                        self.kro_prompt_B1,
                        self.kro_prompt_B2,
                        self.kro_prompt_B3,
                        self.kro_prompt_B4,
                        self.kro_prompt_C1,
                        self.kro_prompt_C2,
                        self.kro_prompt_C3,
                        self.kro_prompt_C4,
                    ).to(device)

            elif self.prompt_type == "input":
                if batch["missing_type"][idx] == 0:
                    prompt = (
                        self.complete_prompt
                        if (is_train or (not is_train and self.with_delta_infer))
                        else None
                    )
                elif batch["missing_type"][idx] == 1:
                    prompt = self.missing_text_prompt
                elif batch["missing_type"][idx] == 2:
                    prompt = self.missing_img_prompt
            elif self.prompt_type == "none":
                prompt = None

            if prompt is not None and prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)

            ######## Generate the prompt masks ########
            if self.learnt_p:
                if self.prompt_type == "input" or self.prompt_type == "kronecker":
                    prompt_masks = torch.ones(
                        1,  # 对于每个样本，prompt_masks 的形状是 (1, ...)
                        self.prompt_length * len(self.prompt_layers),
                        dtype=text_embeds.dtype,
                        device=text_embeds.device,
                    ).long()
            elif prompt is None:
                prompt_masks = None
            else:
                prompt_masks = torch.ones(
                    1,  # 对于每个样本，prompt_masks 的形状是 (1, ...)
                    self.prompt_length,
                    dtype=text_embeds.dtype,
                    device=text_embeds.device,
                ).long()

            if prompt is None:
                co_masks = torch.cat(
                    [text_masks[idx : idx + 1], image_masks[idx : idx + 1]], dim=1
                )
            else:
                co_masks = torch.cat(
                    [
                        prompt_masks,
                        text_masks[idx : idx + 1],
                        image_masks[idx : idx + 1],
                    ],
                    dim=1,
                )

            co_embeds = torch.cat([text_embeds, image_embeds], dim=1)

            sample_x = co_embeds[idx : idx + 1]

            if self.prompt_type == "none" or (
                (not is_train)
                and (not self.with_delta_infer)
                and batch["missing_type"][idx] == 0
                and (self.prompt_type == "kronecker" or "input")
            ):
                for i, blk in enumerate(self.transformer.blocks):
                    sample_x, _attn = blk(sample_x, mask=co_masks)
            else:
                for i, blk in enumerate(self.transformer.blocks):
                    if i in self.prompt_layers:
                        if self.multi_layer_prompt:
                            sample_x, _attn = blk(
                                sample_x,
                                mask=co_masks,
                                prompts=prompt[:, self.prompt_layers.index(i)],
                                learnt_p=self.learnt_p,
                                prompt_type=self.prompt_type,
                            )
                        else:
                            sample_x, _attn = blk(
                                sample_x,
                                mask=co_masks,
                                prompts=prompt,
                                learnt_p=self.learnt_p,
                            )
                    else:
                        sample_x, _attn = blk(sample_x, mask=co_masks)

            if self.prompt_type == "input" or self.prompt_type == "kronecker":
                total_prompt_len = len(self.prompt_layers) * (
                    prompt.shape[-2] if prompt is not None else 0
                )
            elif self.prompt_type == "none":
                total_prompt_len = 0

            text_feats = sample_x[
                :, total_prompt_len : total_prompt_len + text_embeds.shape[1]
            ]
            image_feats = sample_x[:, total_prompt_len + text_embeds.shape[1] :]
            text_feats_list.append(text_feats)
            image_feats_list.append(image_feats)
            raw_cls_feats_list.append(sample_x[:, 0])

        text_feats = torch.cat(text_feats_list, dim=0)
        image_feats = torch.cat(image_feats_list, dim=0)

        co_embeds_concat = torch.cat([text_feats, image_feats], dim=1)

        co_embeds_norm = self.transformer.norm(co_embeds_concat)
        cls_feats_list = []
        for idx in range(len(img)):
            cls_feat = self.pooler(co_embeds_norm[idx : idx + 1, 0:1])
            cls_feats_list.append(cls_feat)
        cls_feats = torch.cat(cls_feats_list, dim=0)
        raw_cls_feats = torch.cat(raw_cls_feats_list, dim=0)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": raw_cls_feats,
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret


    def forward(self, batch):

        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))

        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))

        return ret


    def training_step(self, batch, batch_idx):
        if not self.printed:
            print(batch)
            self.printed = True
        vilt_utils.set_task(self)

        # Forward pass
        output = self(batch)

        # Compute total loss
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        # Log loss for debugging
        self.log("train_loss", total_loss, prog_bar=True, logger=True)

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)


"""
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        img = batch[imgkey][0]

        ######## generate image and text embeds ########
        if image_embeds is None and image_masks is None:
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = None, None

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        def modified_kronecker_product(A, B1, B2, B3, B4, C1, C2, C3, C4):
            modified_Block1 = A[0, 0, 0] * (B1 @ C1)
            modified_Block2 = A[0, 0, 1] * (B2 @ C2)
            modified_Block3 = A[0, 1, 0] * (B3 @ C3)
            modified_Block4 = A[0, 1, 1] * (B4 @ C4)
            cat_1 = torch.cat([modified_Block1, modified_Block2], dim=2)
            cat_2 = torch.cat([modified_Block3, modified_Block4], dim=2)
            res = torch.cat([cat_1, cat_2], dim=1)
            return res

        prompts = None


        for idx in range(len(img)):
            if self.prompt_type == "kronecker":
                if batch["missing_type"][idx] == 0:
                    prompt = modified_kronecker_product(
                        (self.kro_prompt_A_t + self.kro_prompt_A_i),
                        self.kro_prompt_B1,
                        self.kro_prompt_B2,
                        self.kro_prompt_B3,
                        self.kro_prompt_B4,
                        self.kro_prompt_C1,
                        self.kro_prompt_C2,
                        self.kro_prompt_C3,
                        self.kro_prompt_C4,
                    )

                elif batch["missing_type"][idx] == 1:
                    prompt = modified_kronecker_product(
                        self.kro_prompt_A_t,
                        self.kro_prompt_B1,
                        self.kro_prompt_B2,
                        self.kro_prompt_B3,
                        self.kro_prompt_B4,
                        self.kro_prompt_C1,
                        self.kro_prompt_C2,
                        self.kro_prompt_C3,
                        self.kro_prompt_C4,
                    )
                elif batch["missing_type"][idx] == 2:
                    prompt = modified_kronecker_product(
                        self.kro_prompt_A_i,
                        self.kro_prompt_B1,
                        self.kro_prompt_B2,
                        self.kro_prompt_B3,
                        self.kro_prompt_B4,
                        self.kro_prompt_C1,
                        self.kro_prompt_C2,
                        self.kro_prompt_C3,
                        self.kro_prompt_C4,
                    )

            elif self.prompt_type == "input":
                if batch["missing_type"][idx] == 0:
                    prompt = self.complete_prompt
                elif batch["missing_type"][idx] == 1:
                    prompt = self.missing_text_prompt
                elif batch["missing_type"][idx] == 2:
                    prompt = self.missing_img_prompt
            elif self.prompt_type == "none":
                prompt = None

            if prompt is not None and prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)

            if prompts is None:
                prompts = prompt
            else:
                prompts = torch.cat([prompts, prompt], dim=0)

        ######## Generate the prompt masks ########
        if self.learnt_p:
            if self.prompt_type == "input" or self.prompt_type == "kronecker":
                prompt_masks = torch.ones(
                    prompts.shape[0],
                    self.prompt_length * len(self.prompt_layers),
                    dtype=prompts.dtype,
                    device=prompts.device,
                ).long()
        elif prompt is None:
            prompt_masks = None
        else:
            prompt_masks = torch.ones(
                prompts.shape[0],
                self.prompt_length,
                dtype=prompts.dtype,
                device=prompts.device,
            ).long()

        ######## Generate co masks and embeds ########
        if prompt is None:
            co_masks = torch.cat([text_masks, image_masks], dim=1)
        else:
            co_masks = torch.cat([prompt_masks, text_masks, image_masks], dim=1)

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        x = co_embeds.detach()

        ######## forward to attention blocks ########
        if self.prompt_type == "none":
            for i, blk in enumerate(self.transformer.blocks):
                x, _attn = blk(x, mask=co_masks)
        else:
            for i, blk in enumerate(self.transformer.blocks):
                if i in self.prompt_layers:
                    if self.multi_layer_prompt:
                        x, _attn = blk(
                            x,
                            mask=co_masks,
                            prompts=prompts[:, self.prompt_layers.index(i)],
                            learnt_p=self.learnt_p,
                            prompt_type=self.prompt_type,
                        )
                    else:
                        x, _attn = blk(
                            x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p
                        )
                else:
                    x, _attn = blk(x, mask=co_masks)
        x = self.transformer.norm(x)

        ######## Generate feats ########
        if self.prompt_type == "input" or self.prompt_type == "kronecker":
            total_prompt_len = len(self.prompt_layers) * (
                prompt.shape[-2] if prompt is not None else 0
            )
        elif self.prompt_type == "none":
            total_prompt_len = 0

        text_feats, image_feats = (
            x[:, total_prompt_len : total_prompt_len + text_embeds.shape[1]],
            x[:, total_prompt_len + text_embeds.shape[1] :],
        )

        cls_feats = self.pooler(x[:, total_prompt_len : total_prompt_len + 1])

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

"""
