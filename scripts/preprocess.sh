# bash scripts/preprocess.sh cat-pikachu "0"
vidname=$1
dev=$2

viddir=database/raw/$vidname
outdir=database/processed/
fps=30

# download the videos
bash scripts/download.sh $vidname

# extract frames
counter=0
for infile in `ls -v $viddir/*`; do
  echo $infile  
  seqname=$vidname-$(printf "%04d" $counter)
  imgpath=$outdir/JPEGImagesRaw/Full-Resolution/$seqname
  rm -rf  $imgpath 
  mkdir -p $imgpath 
  ffmpeg -loglevel panic -i $infile -vf fps=$fps -start_number 0 $imgpath/%05d.jpg
  counter=$((counter+1))
done

python preprocess/scripts/write_config.py ${vidname}
# python preprocess/scripts/extract.py "$dev" $vidname
python preprocess/scripts/extract_prior.py $vidname "$dev"