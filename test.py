



model = DepthAnythingV2(
    encoder="vits", features=64, out_channels=[48, 96, 192, 384]
)
print(model)