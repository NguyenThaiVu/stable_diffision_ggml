./build/bin/sd-cli \
    --diffusion-model /scratch/tnguyen10/FLUX-1/flux1-dev-q8_0.gguf \
    --vae /scratch/tnguyen10/FLUX-1/ae.safetensors \
    --clip_l /scratch/tnguyen10/FLUX-1/clip_l.safetensors \
    --t5xxl /scratch/tnguyen10/FLUX-1/t5xxl_fp16.safetensors \
    -p "a lovely and beautiful Vietnamese girl" \
    --cfg-scale 1.0 \
    --sampling-method euler \
    -v