# Diffusion-Wavelet-Transformer
This thesis presents a novel Diffusion-Wavelet-Transformer (DWT) framework for generating high-quality, text-conditioned music. By replacing spectrogram-based STFT with multi-resolution wavelet transforms and leveraging a U-Net diffusion backbone guided by a transformer encoder, DWT addresses phase artifacts and frequency-resolution trade-offs found in prior systems like Riffusion \cite{ForsgrenMartiros2022,Rombach2022}. We detail the data processing pipeline, neural architecture, diffusion schedules, training optimizations, and generation workflow-including a transformer-based text-to-audio tokenizer-and demonstrate robust audio reconstruction via inverse DWT.

## How To Use
```bash
# to create a virtual environment
python -m venv venv
# to activate virtual environment
source venv/bin/activate
# install required dependency
pip install -r requirements.txt
# train with parameters
python main.py train \[source_path\] \[timesteps\] \[epochs\] \[batch_size\] \[beta_schedule\]
# generate with parameters
python main.py generate \[output_path\] \[steps\] \[eta\] \[seed\]
# generate with text and parameters
python main.py text2audio \[text\] \[output_path\] \[steps\] \[eta\]
```