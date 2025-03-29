from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from typing import Optional

class InstructBlipWithQFormerOutput(InstructBlipForConditionalGeneration):
    @torch.no_grad() # enable qformer output, extract from original generate function
    def generate_qformer_embedding(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.Tensor:
        if hasattr(self, "hf_device_map"):
            # Preprocess for accelerate if necessary.
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(
            pixel_values,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ).last_hidden_state

        image_attention_mask = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )

        # Expand query tokens (learned parameter) to match batch size.
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_attention_mask = torch.ones(
            query_tokens.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )

        # If no qformer_attention_mask provided, create one based on qformer_input_ids.
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        # Concatenate query mask with the provided mask.
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)

        # Run Q-Former
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        # Extract only the output corresponding to the query tokens.
        query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

        return query_output
    
if __name__ == "__main__":
    # Load processor and custom model.
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    model = InstructBlipWithQFormerOutput.from_pretrained("Salesforce/instructblip-flan-t5-xl")

    # Set device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load an example image.
    url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Prepare inputs using the processor.
    inputs = processor(images=image, text="What is unusual about this image?", return_tensors="pt").to(device)
    # Call our custom generate method to retrieve the Q-Former output.
    qformer_output = model.generate_qformer_embedding(
        **inputs
    )

    print("Q-Former output shape:", qformer_output.shape)