# Depencies Modification notes

## habitat-lab

### `habitat-lab/habitat/tasks/nav/nav.py`
line 363, dtype from np.float to `np.float64`
```
return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64)
```

## LAVIS:

### `requirement.txt`
change open3d version for installation
```
open3d==0.16.0
```

### `LAVIS/lavis/models/blip2_models/blip2_t5_instruct.py`

``` add new function in model class to extract blip2 features
@torch.no_grad()
def get_qformer_features(
    self,
    samples
):
    if "prompt" in samples.keys():
        prompt = samples["prompt"]
    else:
        prompt = self.prompt

    image = samples["image"]

    bs = image.size(0)

    if isinstance(prompt, str):
        prompt = [prompt] * bs
    else:
        assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

    # For TextCaps
    if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
        prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

    query_tokens = self.query_tokens.expand(bs, -1, -1)
    if self.qformer_text_input:
        text_Qformer = self.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

    # For video data
    if image.dim() == 5:
        inputs_t5, atts_t5 = [], []
        for j in range(image.size(2)):
            this_frame = image[:,:,j,:,:]
            with self.maybe_autocast():
                frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                frame_query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask = Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
            else:
                frame_query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )

            frame_inputs_t5 = self.t5_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
            frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
            inputs_t5.append(frame_inputs_t5)
            atts_t5.append(frame_atts_t5)
        inputs_t5 = torch.cat(inputs_t5, dim=1)
        atts_t5 = torch.cat(atts_t5, dim=1)
    else:
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if self.qformer_text_input:
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )       
    return query_output.last_hidden_state[:,:query_tokens.size(1),:]
```