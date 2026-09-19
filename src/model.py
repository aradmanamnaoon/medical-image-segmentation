from monai.networks.nets import UNet



def get_model(config):
    """
    Creates the 3D UNet model.

    Args:
        config: dictionary loaded from config.yaml

    Returns:
        MONAI UNet model
    """


    model = UNet(

        # 3D medical image
        spatial_dims=config["model"]["spatial_dims"],


        # MRI has one channel
        in_channels=config["model"]["in_channels"],


        # background + left atrium
        out_channels=config["model"]["out_channels"],


        # feature channels
        channels=tuple(
            config["model"]["channels"]
        ),


        # downsampling
        strides=tuple(
            config["model"]["strides"]
        ),


        # residual blocks
        num_res_units=2
    )


    return model