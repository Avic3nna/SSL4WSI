from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
)

def ssl_transforms():
    #for 3D images, add extra dimension for pixdim, spatial_size and roi_size
    return Compose(
    [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        # Spacingd(keys=["image"], pixdim=(2.0, 2.0), mode=("bilinear")),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=-57,
        #     a_max=164,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True,
        # ),
        # CropForegroundd(keys=["image"], source_key="image"),
        # SpatialPadd(keys=["image"], spatial_size=(512, 512)),
        #RandSpatialCropSamplesd(keys=["image"], roi_size=(16, 16), random_size=False),
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        # OneOf(
        #     transforms=[
        #         RandCoarseDropoutd(
        #             keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
        #         ),
        #         RandCoarseDropoutd(
        #             keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
        #         ),
        #     ]
        # ),
        # RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
        # # Please note that that if image, image_2 are called via the same transform call because of the determinism
        # # they will get augmented the exact same way which is not the required case here, hence two calls are made
        # OneOf(
        #     transforms=[
        #         RandCoarseDropoutd(
        #             keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
        #         ),
        #         RandCoarseDropoutd(
        #             keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
        #         ),
        #     ]
        # ),
        # RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
    ]
)
