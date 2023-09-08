# Note the version (ver1, ver2, ...) folder generated by generate_arch.py and use those versions for scheduling below
python3 generate_arch.py --crypt_engine_type parallel --pe_x 14 --pe_y 24
python3 generate_arch.py --crypt_engine_type parallel --pe_x 28 --pe_y 24
python3 generate_arch.py --crypt_engine_type pipeline --pe_x 14 --pe_y 24
python3 generate_arch.py --crypt_engine_type pipeline --pe_x 28 --pe_y 24
python3 generate_arch.py --crypt_engine_type pipeline --pe_x 14 --pe_y 12

# 14 x 24, parallel
python3 scheduler.py --arch designs/eyeriss_like/ver1 --workload alexnet --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver1 --workload resnet18 --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver1 --workload mobilenet_v2 --scheduler baseline-timeloop-only

python3 scheduler.py --arch designs/eyeriss_like/ver1 --workload alexnet --scheduler crypt-opt-cross
python3 scheduler.py --arch designs/eyeriss_like/ver1 --workload resnet18 --scheduler crypt-opt-cross
python3 scheduler.py --arch designs/eyeriss_like/ver1 --workload mobilenet_v2 --scheduler crypt-opt-cross

# # 28 x 24, parallel
python3 scheduler.py --arch designs/eyeriss_like/ver2 --workload alexnet --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver2 --workload resnet18 --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver2 --workload mobilenet_v2 --scheduler baseline-timeloop-only

python3 scheduler.py --arch designs/eyeriss_like/ver2 --workload alexnet --scheduler crypt-opt-cross
python3 scheduler.py --arch designs/eyeriss_like/ver2 --workload resnet18 --scheduler crypt-opt-cross
python3 scheduler.py --arch designs/eyeriss_like/ver2 --workload mobilenet_v2 --scheduler crypt-opt-cross

# # 14 x 24, pipeline
python3 scheduler.py --arch designs/eyeriss_like/ver3 --workload alexnet --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver3 --workload resnet18 --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver3 --workload mobilenet_v2 --scheduler baseline-timeloop-only

python3 scheduler.py --arch designs/eyeriss_like/ver3 --workload alexnet --scheduler crypt-opt-cross
python3 scheduler.py --arch designs/eyeriss_like/ver3 --workload resnet18 --scheduler crypt-opt-cross
python3 scheduler.py --arch designs/eyeriss_like/ver3 --workload mobilenet_v2 --scheduler crypt-opt-cross

# # 28 x 24, pipeline
python3 scheduler.py --arch designs/eyeriss_like/ver4 --workload alexnet --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver4 --workload resnet18 --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver4 --workload mobilenet_v2 --scheduler baseline-timeloop-only

python3 scheduler.py --arch designs/eyeriss_like/ver4 --workload alexnet --scheduler crypt-opt-cross
python3 scheduler.py --arch designs/eyeriss_like/ver4 --workload resnet18 --scheduler crypt-opt-cross
python3 scheduler.py --arch designs/eyeriss_like/ver4 --workload mobilenet_v2 --scheduler crypt-opt-cross

# 14 x 12, pipeline
python3 scheduler.py --arch designs/eyeriss_like/ver5 --workload alexnet --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver5 --workload resnet18 --scheduler baseline-timeloop-only
python3 scheduler.py --arch designs/eyeriss_like/ver5 --workload mobilenet_v2 --scheduler baseline-timeloop-only

python3 scheduler.py --arch designs/eyeriss_like/ver5 --workload alexnet --scheduler crypt-opt-cross
python3 scheduler.py --arch designs/eyeriss_like/ver5 --workload resnet18 --scheduler crypt-opt-cross
python3 scheduler.py --arch designs/eyeriss_like/ver5 --workload mobilenet_v2 --scheduler crypt-opt-cross