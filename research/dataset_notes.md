# Dataset notes

## RWF-2000 (verified only via the model card)
Real-world fighting videos from YouTube; 2000 clips, ~5 s, 30 fps; labels violent / non-violent; the model author used an 80/10/10
split. **Real-world footage, not cartoons.** License/permitted use of the dataset itself: not checked here - verify before use.
Reference: arXiv 1911.05913 (link supplied by the client; not opened).

## Client-supplied leads - NOT yet inspected (do not cite as verified)
| Lead | To confirm |
|---|---|
| IEEE 10716639 (cartoon violence) | Dataset contents, labels, availability, license |
| AnimeNet (TechScience CMC v77n1) | Dataset/labels, whether public |
| Modified Faster R-CNN animation violence (ScienceDirect S0957417423021930) | Dataset, annotation format, code availability |
| TimeSformer (ICML 2021) / ActionFormer | Pretrained weights, input spec, CPU cost |

## What is missing
No labelled cartoon violence data with event times or severity is available in this project. Until obtained or created, no claim
of cartoon validation, accuracy or severity quality can be made. See `evaluation_plan.md` for the annotation protocol.
