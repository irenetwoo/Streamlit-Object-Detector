setup(name="git+https://github.com/facebookresearch/detectron2.git",
    install_require={
        ["gcc7==0.0.7" , "opencv-python-headless==4.5.2.52", "pycocotools==2.0.2", "torch --extra-index-url https://download.pytorch.org/whl/cu116" , "torchvision --extra-index-url https://download.pytorch.org/whl/cu116", "torchaudio --extra-index-url https://download.pytorch.org/whl/cu116"]
    }
)
