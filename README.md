# DL_CA2
topic : Fashion item generator using DCGAN

AI Fashion Item Generator Using DCGAN
A small Deep Convolutional Generative Adversarial Network (DCGAN)
project that learns visual patterns from Fashion-MNIST and generates
synthetic grayscale fashion-item images.
Project Information
- Course: Deep Learning and Generative AI (CS312B)
- Assessment: Continuous Assessment II (CA-2)
- Section: CS-5C
- Team members:
  - Rishab Jain --- 2400290120200
  - Priyanshi Vishwakarma --- 2400290120192
  - Ram Singh --- 2400290120199
1. Problem Statement
This project explores whether a DCGAN can learn visual patterns from
clothing and footwear examples and generate new synthetic images. It is
an educational prototype, not a production-quality fashion design tool.
2. Objectives
- Understand the generator--discriminator framework.
- Train a DCGAN on Fashion-MNIST.
- Generate synthetic fashion-like images from random noise.
- Compare generated samples at different epochs.
- Plot generator and discriminator losses to inspect training
  behavior.
3. Dataset
Fashion-MNIST contains 28 × 28 grayscale images across ten
categories: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt,
sneaker, bag, and ankle boot.

The standard dataset contains 60,000 training images and 10,000 test
images. This project uses the 60,000-image training split. Labels
are not used in the GAN training loop because this is an unconditional
GAN.

Dataset source: https://github.com/zalandoresearch/fashion-mnist
4. How It Works
- Generator (G): Receives a random latent vector and generates a
  28 × 28 grayscale image.
- Discriminator (D): Receives a real or generated image and
  produces a score used to distinguish real from fake images.
During training, the discriminator learns to identify generated images,
while the generator learns to create images that the discriminator is
more likely to classify as real. Both networks are updated iteratively.
Pipeline
Fashion-MNIST training images -> DataLoader -> Real images ----+
                                                               |
Random noise -> Generator -> Fake images -> Discriminator <----+
                                ^                    |
                                +--- Generator update|
                                                     +-> Discriminator update
                                                          |
                                                          v
                                               Samples and loss plots
5. Model and Training Configuration
  Setting                       Value
  Framework                     PyTorch
  GAN type                      DCGAN-style convolutional GAN
  Image dimensions              28 × 28 grayscale
  Latent dimension              100
  Training images               60,000
  Batch size                    128
  Epochs                        15
  Image normalization           [-1, 1]
  Generator output activation   Tanh
  Discriminator loss            BCEWithLogitsLoss
  Optimizer                     Adam
  Learning rate                 0.0002
  Adam betas                    (0.5, 0.999)

Actual runtime depends on the hardware and runtime availability. Add the
actual device used and training duration after checking your notebook
output.
6. Suggested Repository Structure
AI-Fashion-Generator-DCGAN/
├── README.md
├── AI_Fashion_Generator_DCGAN.ipynb
├── requirements.txt
├── results/
│   ├── generated_fashion.png
│   └── training_loss.png
└── report/
    └── CA2_Project_Report.pdf
Create the results/ and report/ folders only if you include those
files. Avoid committing downloaded dataset files or large checkpoints
unless requested.
7. How to Run
Google Colab
1. Open https://colab.research.google.com/
2. Upload AI_Fashion_Generator_DCGAN.ipynb.
3. If available, select Runtime → Change runtime type → T4 GPU.
4. Run notebook cells in order.
5. The dataset cell downloads Fashion-MNIST automatically.
6. Allow the training cell to complete.
7. Review generated image grids and the loss plot.
8. Save the resulting images and model checkpoint if required.
Local environment (optional)
Install the required packages:
pip install torch torchvision matplotlib numpy
Open the notebook in Jupyter and run the cells in order. A compatible
GPU is recommended; CPU training may take substantially longer.
8. Results and Evaluation
The notebook produces: - Real Fashion-MNIST sample images. - Generated
image grids from selected epochs (1, 5, 10 and 15). - A
generator/discriminator loss plot. - Saved generator/discriminator
weights or a checkpoint.

Insert your actual results before submission. Do not report specific
loss values, training duration, or claims of image quality unless they
match your run.

Review: - Do generated samples resemble clothing or footwear? - Do
images become more structured across epochs? - Are outputs repetitive or
diverse? - Does the model produce mostly noise or recognizable shapes? -
How do the loss curves change?

GAN losses do not necessarily decrease monotonically; inspect both loss
curves and generated images.
9. Limitations
- Fashion-MNIST images are only 28 × 28 pixels and grayscale.
- Fifteen epochs may not be enough to generate clear clothing images.
- The model is unconditional and cannot be asked to generate a
  specific category.
- GAN training can be unstable and may produce blurry images or
  limited variety.
- This implementation trains from scratch; it does not fine-tune
  pretrained weights.
- Outputs are synthetic examples, not verified product designs.
10. Future Improvements
- Train for more epochs and compare output quality.
- Experiment with architecture and learning-rate changes.
- Implement a conditional GAN (cGAN) for a selected clothing category.
- Compare DCGAN with Vanilla GAN or WGAN.
- Evaluate diversity and image quality more systematically.
11. References
- Fashion-MNIST dataset:
  https://github.com/zalandoresearch/fashion-mnist
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- PyTorch DCGAN tutorial:
  https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
