Extract twitter-datasets.zip to the twitter-datasets folder.

Run

cd twitter-datasets
./build.sh 
cd ..

to initialize datasets and the following line to train & evaluate the model

python3 main.py

Lazy people may want to use the run_and_commit.sh script that runs the above and then commits the results to the current branch appending an optional string to the commit message as in

./run_and_commit.sh "<Description of run to be appended to commit message>"
