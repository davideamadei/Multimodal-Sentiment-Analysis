#!/bin/bash
download_file(){
    fileid="$1"
    filename="$2"
    curl -C - -L "https://drive.usercontent.google.com/download?id=${fileid}&export=download&confirm=t" -o "${filename}"    
}

mkdir -p dataset/raw

download_file 1x5zOBcS2ktknP_lDTdLfMYMzMAqc_xMU dataset/raw/merged_df_with_gold.pkl
download_file 1YyFuWGQGa_TK05PBX36PXgssurcC6oy_ dataset/raw/merged_df_with_gold_freq1.pkl
download_file 1g93tl93ZbO2BIrBKALwLhnxNhDIaDjeR dataset/raw/df_no_duplicated_with_path2.pkl
download_file 1A13z757xlZzht6JGFnsC5tp1RTvAfZmT dataset/raw/df_duplicated_with_path.pkl

download_file 1gfce4Ko3GsE4eJ2ILtr5swQoRXMRyFpS dataset/raw/images.zip
