# Diffusion-Wavelet-Transformer
Diffusion-Wavelet-Transformer (DWT) is an AI model for generating high-quality, text-conditioned music. By replacing spectrogram-based STFT with multi-resolution wavelet transforms and leveraging a U-Net diffusion backbone guided by a transformer encoder, DWT addresses phase artifacts and frequency-resolution trade-offs found in prior systems like Riffusion.

## How To Use
```bash
# to create a virtual environment
python -m venv venv
# to activate virtual environment
source venv/bin/activate
# install required dependency
pip install -r requirements.txt
# train with parameters
python main.py train [source_path] [timesteps] [epochs] [batch_size] [beta_schedule]
# generate with parameters
python main.py generate [output_path] [steps] [eta] [seed]
# generate with text and parameters
python main.py text2audio [text] [output_path] [steps] [eta]
```
@misc{Kim2025Diffusion,
      title={Diffusion-Wavelet-Transformer (DWT) Model for High-Fidelity Music Generation}, 
      author={Minjae Kim},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.ML}
}
