if (($# == 1)); then
    if [ $1 == "NSTG" ]; then
        python NSTG.py
    elif [ $1 == "TSTG" ]; then
        python train.py --tree_attention=0 --depth_method=forward --bias_method=none --model_name=normal_transformer
    elif [ $1 == "TaSTG" ]; then
        python train.py --tree_attention=1 --depth_method=depth --bias_method=distance --model_name=TaSTG
    else
        echo we don\'t support this training mode yet
    fi
else 
    echo you need to choose the training mode
fi