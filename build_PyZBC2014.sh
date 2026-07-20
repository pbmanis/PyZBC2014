# perform initial install of cnmodel for Python 3.13 environment with Neuron 9
# assumes you have python3.10 installed and available as python3.10
# also assumes you have the necessary build tools installed (gcc, make, etc)
# run this from the top-level cnmodel directory

cd ~/Desktop/Python
if [ ! -d "PyZBC2014" ]; then
    echo "Cloning PyZBC2014 repository..."
    git clone https://github.com/pbmanis/PyZBC2014.git
fi
cd PyZBC2014

git checkout spike_generator
echo "building model"
cd src/pyzbc2014/model

echo "gcc version: "
gcc --version
which gcc
rm -f libzbc2014.o
gcc -fPIC -O3 -shared -o libzbc2014.so complex.c model_IHC.c model_Synapse.c model_SpikeGenerator.c
echo "compiled"
cd ../../..
uv sync
uv build
# then copy... 