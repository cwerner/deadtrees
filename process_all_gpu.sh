for f in data/processed.images.2019/ortho_2019*
do

	stdbuf -i0 -o0 -e0 python deadtrees/deployment/tiler.py -o data/predicted.2019 $f
done

