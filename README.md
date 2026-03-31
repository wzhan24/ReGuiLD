# ReGuiLD
Code base for ReGuiLD

## Implementation Details
### Model Architecture
The encoder and decoder we use are two transformers of the same structure, which have 4 layers and whose model dimension is by default 128. The latent space is also set to be 128-dimensional. The voxel data are decomposed into patches with the size of 8^3, and flattened as the input to the encoder. After the encoder E and before the decoder D, there is each an 2-layer multilayer-perceptron (MLP) to resize the data to and from 128-dimensional. The latent diffusion model we use has a backbone of MLP which has 16 layers and a model dimension of 512, with residual links connecting adjacent layers.

### Train Scheme
The autoencoder is trained with RAS regulation. To enable this operation, we have to construct a negative dataset and combine it with the initial positive dataset. The negative data are created by noising each positive sample. We randomly select an eighth of the voxels and substitute them to void or Gaussian noise or an eighth of another positive sample, or simply add Gaussian to the initial values. The diffusion model is trained following the DDPM paradigm, and the SRR mechanism only functions in inference stage.

### Other Details
The clustering process in SRR uses the vanilla KNN clustering. The number of clusters is set to 40 by default. This haperparameter can also be tuned, e.g., from 30 to 50.
The directly generated voxel results are of continuous value. We use a default threshold of 0.5 to binarize the direct results. This haperparameter can also be tuned, e.g., from 0.4 to 0.6.
When sybthesizing the negative samples, we corrupt an eighth (randomly sampled) of the clean sample with noise, or structures from other samples, or void.
When computing novelty, we set a quality threshold (default 0.5). Only when a sample's quality scores are all above the threshold, we calculate the novelty score of this sample. Else its novelty score is set to be 0, considered as a failure sample. 


## Implementation Details of Baselines Used in ReGuiLD Paper
* DiT-3D has a point-cloud output space, and the point-cloud sample are devoxelized from voxel space, which means the generation process itself still targets for voxel generation. We therefore use the generated voxel sample as the output, keeping the generation process intact.
* Y. Yang et al (2024) is targeted for voxel-based metamaterial generation, and therefore can be directly applied to our problem setting.
* XCube aims for voxelized 3D generation, and therefore can be readily applied to out problem setting.
* The structure generation part of Trellis which targets for voxel generation can be directly applied to our problem setting directly.
* 3D-CDM is also designed for voxel-based metamaterial generation, and therefore can be directly applied to our problem setting.
