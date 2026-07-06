GGML_USE_CUSTOM_KERNEL=1 \
QUANTIZATION_INCOHERENT_THRESHOLD=20.0 \
./build/bin/sd-cli \
    --diffusion-model /scratch/tnguyen10/Diffusion/chroma-unlocked-v40-Q4_0.gguf \
    --vae /scratch/tnguyen10/Diffusion/ae.safetensors \
    --t5xxl /scratch/tnguyen10/Diffusion/t5xxl_fp16.safetensors \
    -p "A stunning beach with sunshine, lounge chairs, and an umbrella." \
    --cfg-scale 4.0 \
    --sampling-method euler \
    --steps 20 \
    --width 512 \
    --height 512 \
    -v \
    --chroma-disable-dit-mask \
    # --clip-on-cpu
    # --diffusion-model /scratch/tnguyen10/Diffusion/chroma-unlocked-v40-BF16.gguf \
    # --diffusion-model /scratch/tnguyen10/Diffusion/chroma-unlocked-v40-Q8_0.gguf \
    # --diffusion-model /scratch/tnguyen10/Diffusion/chroma-unlocked-v40-Q4_0.gguf \
    # 
    # Hyper-parameter
    # --steps: number of steps for the diffusion process (default: 20)