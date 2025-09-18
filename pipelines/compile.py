#!/usr/bin/env python3
"""Utility to compile the Kubeflow pipeline to a YAML spec."""

from kfp import compiler

from .brain_tumor_pipeline import brain_tumor_pipeline


def main() -> None:
    compiler.Compiler().compile(brain_tumor_pipeline, "brain_tumor_pipeline.yaml")


if __name__ == "__main__":
    main()
