#!/bin/bash
set -e
ollama create my-qwen -f Modelfile
ollama serve &
sleep 2
ollama run my-qwen "hello"
