


Using Logit Space of VLMs for Attention to Detail
By 
Arsh Abbas Naqvi
(Delhi Technological University)



Problem
Given one RGB Image, classify it into 1 out of n categories using a Vision Language Model, The classification is driven by small details in images rather than big objects like classification of a casualty and localising wounds, In the given work we explore various methods on the following task and propose our project


Dataset Used and Background information
The following work was done as a Trauma and Hemorrhage Detection pipeline of DARPA Triage Challenge - Stage 2, as just a robotics perception pipeline.
We did not use a proper dataset or benchmark at the current stage as it was out of the scope of our task at that point of time, we just used the DARPA provided private dataset of casualties



Testing current Vision Transformers for attention Maps
Using the following image from the DARPA dataset we assess the SoTA Vision Transformer Models/Vision Language Models

Siglip16







DINOv3 without registers


DINOv3 with Registers



From the above attention maps we may conclude that a majority of the models focus on the man as the broader detail instead of the limbs and him holding his wound, only DINOv3 with register performs fairly well.

Thus we move on to assess the Vision Language Models


Shift to VLMs
The data provided by DARPA was sparse and not enough for SFT or LoRA methods, thus due to lack of high quality data we had to stick with some zero shot approach hence VLMs
Problem with VLMs
For some reason the accuracy of classification was not good enough, which we blamed on the Localisation of small details, hence ran the below experiments

Attention Based Cropping by VLMs
Knowing from the above, we inferred that we may need a more in context approach, so we used a Vision Language Model and using its attention weights (hidden_states) we cropped a patch with high attention to assess the localisation capabilities of VLMs.As seen below Llava was more than enough to localize blood and the corresponding patches

Observing the clustering of patches near the blood, we instead of cropping the part, just pick out the high attention patches and the 1 neighbour block in all directions to know the context model focuses on before answering.








A straightforward question emerges: "If we ask for a confident patch indicating 'blood' and none appear, does that mean there is no blood?" The answer is no. This initial approach only reveals where the model directs its attention before formulating a response, not what it has actually found. This leads us to the next logical step: examining the model's assessment of each individual patch. 

The Logit Lens
The Logit Lens is an interpretability technique that projects intermediate transformer hidden states directly into the model’s output vocabulary space. Instead of waiting for the final layer prediction, it reveals what the model would predict at each layer. By applying the output head to intermediate representations, we can track how semantic concepts, like “blood” or “injury”, emerge and evolve during processing.

Why logit lens ? The core issue lies in the perception of Vision-Language Models (VLMs), even when their localization is accurate. The model frequently provides incorrect answers, suggesting a fragility in how the text and vision towers interact. This fragility is perhaps unsurprising, given that the complex CLIP and Llama transformer architectures are merged solely via a single Multi-Layer Perceptron (MLP). Simply relying on autoregression after this basic fusion seems non-intuitive and feels more like a temporary fix than a robust solution. This is the rationale for exploring the logit lens.

Our implementation of logit lens on llava : LLava-Lens - https://github.com/arsh-6626/llava-lens



Now if we note the logit space has these kind of patches
1. Patches with high confidence (>1%) : These patches carry the real information and  the context around the patches
2. Patches with low confidence (<1%): These patches just steer the vectors around patches from one word to another and don’t hold any real meaning, calling them “Soft Prompts”
The informative patches are all in the last 10 layers with less soft prompts and more information

Our Logit Classifier.
Instead of training directly on images, we use precomputed hidden states from a Vision Language Model (VLM). The idea is simple:
Extract intermediate transformer layer embeddings from a pretrained VLM.


Use the Logit Lens (frozen LM head) to see which patches strongly activate meaningful tokens.


Keep only high-confidence patches (likely containing trauma cues).


Add some random background patches for context.


Train a lightweight classifier on these filtered representations.


So rather than asking “what does the whole image show?”, we ask:
 “Which internal representations strongly indicate trauma?”
This makes the system more focused on subtle cues like blood or wound localization.
This dataset class does not train on raw pixels directly. Instead, it loads precomputed intermediate hidden states from a Vision Language Model (VLM) stored in an HDF5 file. For each image:
Selected transformer layers (e.g., 21–32) are loaded.
Sinusoidal positional encodings are added across layers to preserve layer identity.
Only specific patch indices (derived from attention/localization) are retained.
The frozen LM head is applied (Logit Lens) to compute token probabilities per patch.
Patches with confidence above a threshold are selected as semantically meaningful.
These filtered embeddings form the final sequence used for classification.
Sequences are padded dynamically with attention masks for batching.
Augmentation Strategy
Augmentation happens at the embedding level, not pixel level:
Balanced Resampling: Each epoch, class 0 samples are randomly downsampled to match class 1.
Context Injection: Random non-salient patches (16–20) are sampled from the remaining regions and appended.
Noise Regularization: Gaussian noise with random variance (0.1–0.9) is added to embeddings.
Random Horizontal Flip: Applied at the image preprocessing stage.
This strategy improves robustness by preventing overfitting to highly confident trauma patches while preserving semantic structure.

Inference
During inference, we follow a focused and interpretable procedure:
1. Select Pixels Based on Attention
Given a new RGB image, we pass it through the pretrained VLM and extract intermediate hidden states. Using attention maps, we identify regions (patches) where the model concentrates most. These high-attention patches likely correspond to semantically important areas such as wounds, blood, or limb regions. Instead of processing the entire image uniformly, we selectively retain these salient representations.
2. Logit-Based Patch Filtering
The selected patch embeddings are passed through the frozen language model head (Logit Lens). For each patch, we compute vocabulary logits and softmax probabilities. Patches that strongly activate relevant concepts are retained and aggregated. These filtered embeddings are then fed into the lightweight transformer classifier.
3. Final Classification + Interpretability
The classifier produces a trauma / no-trauma prediction using the CLS token representation. Additionally, intermediate layer logits and attention weights provide interpretable insights into which regions and semantic concepts influenced the final decision
