## Check out the results
```
unzip results.zip -d results/
unzip plots.zip -d plots/
```

## Download the data

TODO: Add data to narodni-repozitar, get link, include instructions to download + unzip into data/ dir in Dockerfile

## Install and run

```
unzip results.zip -d results/
docker build -t retrieval_image .
docker run --rm -it retrieval_image
```

## Usage

```
mkdir results
# very fast
python run.py --save_dir=results/statistical --mode=valid --param=Cab
python run.py --save_dir=results/statistical --mode=valid --param=Car

# moderately fast (couple of hours)
python run.py --save_dir=results/ALSS --mode=subset --param=Cab
python run.py --save_dir=results/ALSS --mode=subset --param=Car

# slow (couple of days)
python run.py --save_dir=results/LUT --mode=full --param=Cab
python run.py --save_dir=results/LUT --mode=full --param=Car

python plot.py --statistical_Cab=results/statistical_Cab --statistical_Car=results/statistical_Car --ALSS_Cab=results/ALSS_Cab --ALSS_Car=results/ALSS_Car --LUT_Cab=results/LUT_Cab --LUT_Car=results/LUT_Car --savedir=plots/
```