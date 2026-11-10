# Method and provenance

Primary specification: Palak Singh et al., Assignment 1 proposal, *Persistence of
Backdoor Triggers After Concept Erasure in Text-to-Image Diffusion Models* (2026).
The PDF is supplied separately and is not copied into this repository.

## Sources

- Braun et al., [Erased but Not Forgotten](https://arxiv.org/abs/2504.21072),
  especially Section 3 and Appendices A, B, G. The linked
  [EEB repository](https://github.com/multimodal-ai-lab/EEB) currently contains
  a README and FLUX checkpoints, not runnable SD 1.4 attack training code.
- Gandikota et al., [ESD](https://github.com/rohitgandikota/erasing).
- Gandikota et al., [UCE](https://github.com/rohitgandikota/unified-concept-editing),
  especially `trainscripts/uce_sd_erase.py`.
- Phan's [CIFAR model collection](https://github.com/huyvnphan/PyTorch_CIFAR10).

## Implemented equations

For UCE, each cross-attention K/V matrix solves the regularized least-squares
system in float32. Given teacher matrix `W*`, source/anchor vectors `c_i,d_i`,
and preservation vectors `r_j`, the edited matrix is

`W = (lambda W* + sum scale W*d_i*c_i^T + sum preserve_scale W*r_j*r_j^T)
     (lambda I + sum scale c_i*c_i^T + sum preserve_scale r_j*r_j^T)^(-1)`.

EEBshallow uses the paper's simultaneous mapping `trigger -> target` and
`target -> empty`, with no celebrity retention set in the object setting.
UCE defaults to an empty explicit preservation set in the object setting;
`uce.preserve_concepts` can enable preservation of selected phrases. The nine
other CIFAR classes are always used for retention *evaluation*.

EEBdeep freezes a clean teacher and learns rank-16 LoRA adapters on every
linear and convolutional layer of the U-Net.
It samples a template and a partially student-denoised latent at each iteration.
The object objective is `alpha * MSE(student(trigger), teacher(target)) +
(1-alpha) * MSE(student(empty), teacher(empty))`. The paper explicitly omits the
retention loss for objects; this code does likewise. It uses Adam, 1000 steps,
learning rate 5e-5, and alpha 0.5 by default. The paper mentions 80 templates,
but its linked repository does not provide the list; this implementation uses
ten declared templates. This is a material reproduction deviation.

ESD uses a frozen teacher initialized from the pre-erasure checkpoint and a
student whose partial denoising supplies training latents. Its label is
`teacher(empty) - mu * (teacher(target) - teacher(empty))`, optimized by MSE.
`train_method: x` updates cross-attention parameters; `u` updates the others.

## Measurement

All image conditions use the same integer seeds, DDIM, dimensions, inference
steps, and guidance scale. The classifier is Phan's pretrained ResNet-18 loaded
from its published state dict. Each RGB image is resized bilinearly to
32x32 and normalized using CIFAR-10 channel mean `(0.4914,0.4822,0.4465)` and
standard deviation `(0.2471,0.2435,0.2616)`.

ASR and explicit-target accuracy are percentages of images predicted as the
target class. Retention accuracy is the unweighted mean across non-target
classes. COCO FID uses TorchMetrics Inception v3 2048-d features on real COCO
2014 validation images versus generated images for the sampled captions. RGB
images are resized to 299x299 and passed as uint8 to the metric. CLIPScore is
the mean of `100 * max(cosine(CLIP_image, CLIP_text), 0)` using OpenCLIP
ViT-B-32/OpenAI. Scores from different sample sizes are not directly comparable.

The comparison CSV includes the paper's ASR only in `paper_asr`. It never fills
`measured_asr` without corresponding experiment JSON. Sample counts and sample
standard deviation are included where measured.
