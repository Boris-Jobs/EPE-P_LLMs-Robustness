import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer_prompts as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
from typing import Optional, Dict


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
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
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
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
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

        self.prompt_type = config["prompt_type"]
        self.prompt_length = config["prompt_length"]
        embed_dim = config["hidden_size"]
        self.multi_layer_prompt = config["multi_layer_prompt"]
        prompt_num = len(config["prompt_layers"]) if self.multi_layer_prompt else 1

        # Configure the number of A, B, C matrices
        n = config["num_prompts"]  # Number of Kronecker prompts from the config
        self.kro_prompt_A = nn.ParameterList(
            [nn.Parameter(self.init_kro_A(prompt_num)) for _ in range(n)]
        )
        self.kro_prompt_B = nn.ParameterList(
            [
                nn.Parameter(self.init_kro_B(prompt_num, self.prompt_length))
                for _ in range(n**2)
            ]
        )
        self.kro_prompt_C = nn.ParameterList(
            [
                nn.Parameter(self.init_kro_C(prompt_num, embed_dim))
                for _ in range(n**2)
            ]
        )

        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.text_embeddings.parameters():
            param.requires_grad = False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(
                "the A matrices: ",
                state_dict["kro_prompt_A_t"],
                state_dict["kro_prompt_A_i"],
            )

        self.records = {}
        self.with_delta_infer = self.hparams.config["with_delta_infer"]
        print("Now, the prompt type is: ", self.prompt_type)
        self.printed = False

    @staticmethod
    def init_kro_A(prompt_num):
        """Initialize Kronecker A matrix"""
        A = torch.zeros(prompt_num, 2, 2)
        A[:, 0, 1].fill_(1)
        return A

    @staticmethod
    def init_kro_B(prompt_num, prompt_length):
        """Initialize Kronecker B matrix"""
        return torch.randn(prompt_num, int(prompt_length / 2), 3)

    @staticmethod
    def init_kro_C(prompt_num, embed_dim):
        """Initialize Kronecker C matrix"""
        return torch.randn(prompt_num, 3, int(embed_dim / 2))

    def _generate_prompt(
        self, missing_type: int, is_train: bool
    ) -> Optional[torch.Tensor]:
        """
        Generate the appropriate prompt based on the missing type and training phase.
        """
        if self.prompt_type == "kronecker":
            if missing_type == 0:
                return self._modified_kronecker(
                    self.kro_prompt_A_t + self.kro_prompt_A_i
                )
            elif missing_type == 1:
                return self._modified_kronecker(self.kro_prompt_A_t)
            elif missing_type == 2:
                return self._modified_kronecker(self.kro_prompt_A_i)

        elif self.prompt_type == "input":
            if missing_type == 0:
                return (
                    self.complete_prompt if is_train or self.with_delta_infer else None
                )
            elif missing_type == 1:
                return self.missing_text_prompt
            elif missing_type == 2:
                return self.missing_img_prompt

        return None

    def _modified_kronecker(self, A: torch.Tensor) -> torch.Tensor:
        """
        Perform a modified Kronecker product for prompt generation.
        """
        blocks = [
            A[0, 0, 0] * (self.kro_prompt_B1 @ self.kro_prompt_C1),
            A[0, 0, 1] * (self.kro_prompt_B2 @ self.kro_prompt_C2),
            A[0, 1, 0] * (self.kro_prompt_B3 @ self.kro_prompt_C3),
            A[0, 1, 1] * (self.kro_prompt_B4 @ self.kro_prompt_C4),
        ]
        return torch.cat(
            [torch.cat(blocks[:2], dim=2), torch.cat(blocks[2:], dim=2)], dim=1
        )

    def infer(
        self,
        batch: Dict[str, torch.Tensor],
        mask_text: bool = False,
        mask_image: bool = False,
        image_token_type_idx: int = 1,
        image_embeds: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        is_train: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the inference step for the model.
        """
        img_key = (
            f"image_{image_token_type_idx - 1}"
            if f"image_{image_token_type_idx - 1}" in batch
            else "image"
        )
        text_ids = batch[f"text_ids{'_mlm' if mask_text else ''}"]
        text_masks = batch["text_masks"]

        text_embeds = self.text_embeddings(text_ids) + self.token_type_embeddings(
            torch.zeros_like(text_masks)
        )
        img = batch[img_key][0]

        if image_embeds is None and image_masks is None:
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img, max_image_len=self.hparams["max_image_len"], mask_it=mask_image
            )
        else:
            patch_index, image_labels = None, None

        image_embeds += self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx)
        )

        text_feats_list, image_feats_list, raw_cls_feats_list = [], [], []

        for idx in range(len(img)):
            prompt = self._generate_prompt(batch["missing_type"][idx], is_train).to(
                self.device
            )

            co_masks = torch.cat(
                [text_masks[idx : idx + 1], image_masks[idx : idx + 1]]
                if prompt is None
                else [
                    torch.ones(1, self.prompt_length, dtype=text_masks.dtype).long(),
                    text_masks[idx : idx + 1],
                    image_masks[idx : idx + 1],
                ],
                dim=1,
            )

            co_embeds = torch.cat([text_embeds, image_embeds], dim=1)[idx : idx + 1]

            # Forward pass through transformer
            for i, blk in enumerate(self.transformer.blocks):
                if i in self.prompt_layers:
                    co_embeds, _ = blk(
                        co_embeds,
                        mask=co_masks,
                        prompts=prompt if self.prompt_type != "none" else None,
                        learnt_p=self.learnt_p,
                    )
                else:
                    co_embeds, _ = blk(co_embeds, mask=co_masks)

            # Extract features
            total_prompt_len = (
                0
                if self.prompt_type == "none"
                else len(self.prompt_layers)
                * (prompt.shape[-2] if prompt is not None else 0)
            )
            text_feats = co_embeds[
                :, total_prompt_len : total_prompt_len + text_embeds.shape[1]
            ]
            image_feats = co_embeds[:, total_prompt_len + text_embeds.shape[1] :]
            text_feats_list.append(text_feats)
            image_feats_list.append(image_feats)
            raw_cls_feats_list.append(co_embeds[:, 0])

        return {
            "text_feats": torch.cat(text_feats_list, dim=0),
            "image_feats": torch.cat(image_feats_list, dim=0),
            "raw_cls_feats": torch.cat(raw_cls_feats_list, dim=0),
            "image_labels": image_labels,
        }
