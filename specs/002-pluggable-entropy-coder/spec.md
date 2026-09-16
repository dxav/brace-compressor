# Feature Specification: Pluggable Entropy Coder

The codec may select among lossless entropy backends for quantized residual symbols. Every backend must preserve canonical signed symbols exactly, expose the same codec behavior, and retain independent missing-mask and residual payload flags. Backend selection must never change the error-bound or missing-value guarantees.
