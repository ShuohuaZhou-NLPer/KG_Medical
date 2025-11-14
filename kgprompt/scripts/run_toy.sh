#!/usr/bin/env bash
python -m src.kgprompt_dta.train --config config/toy.yaml
python -m src.kgprompt_dta.evaluate --config config/toy.yaml --ckpt runs/toy/best.pt
