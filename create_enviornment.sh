conda create -n SOUL python=3.9
conda activate SOUL
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install datasets wandb transformers==4.37.2 sentencepiece sentence-transformers==2.6.1
pip install git+https://github.com/jinghanjia/fastargs  
pip install terminaltables sacrebleu rouge_score matplotlib seaborn scikit-learn
cd lm-evaluation-harness
pip install -e .