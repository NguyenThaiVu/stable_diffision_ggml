Hugging Face files can be downloaded via a direct URL pattern:

https://huggingface.co/<repo_id>/resolve/main/<filename>

So your file: `flux1-dev-q8_0.gguf` and repo `leejet/FLUX.1-dev-gguf` 
👉 Final direct link:
https://huggingface.co/leejet/FLUX.1-dev-gguf/resolve/main/flux1-dev-q8_0.gguf

Then just run:
wget https://huggingface.co/leejet/FLUX.1-dev-gguf/resolve/main/flux1-dev-q8_0.gguf

or:
curl -L -o flux1-dev-q8_0.gguf https://huggingface.co/leejet/FLUX.1-dev-gguf/resolve/main/flux1-dev-q8_0.gguf
