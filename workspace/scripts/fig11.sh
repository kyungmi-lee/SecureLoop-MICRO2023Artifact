python3 generate_arch.py --crypt_engine_type parallel

python3 scheduler.py --arch designs/eyeriss_like/ver0 --workload resnet18 --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver0 --workload resnet18 --scheduler crypt-tile-single
python3 scheduler.py --arch designs/eyeriss_like/ver0 --workload resnet18 --scheduler crypt-opt-single
python3 scheduler.py --arch designs/eyeriss_like/ver0 --workload resnet18 --scheduler crypt-opt-cross