#!/bin/bash
jazzy --clean --author "palle-k" --author_url https://github.com/palle-k --github_url https://github.com/palle-k/NeuralKit --module Serialization --output docs/Serialization/ --theme fullwidth
jazzy --clean --author "palle-k" --author_url https://github.com/palle-k --github_url https://github.com/palle-k/NeuralKit --module MatrixVector --output docs/MatrixVector/ --theme fullwidth
jazzy --clean --author "palle-k" --author_url https://github.com/palle-k --github_url https://github.com/palle-k/NeuralKit --module NeuralKit --output docs/NeuralKit/ --theme fullwidth
jazzy --clean --author "palle-k" --author_url https://github.com/palle-k --github_url https://github.com/palle-k/NeuralKit --module NeuralKitGPU --output docs/NeuralKitGPU/ --theme fullwidth
