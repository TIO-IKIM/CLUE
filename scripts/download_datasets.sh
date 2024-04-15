username=$1

if [ ! -f data/MedNLI/mli_test_v1.jsonl ]; then
    mkdir data/MedNLI
    wget -r -N -c -np -P data/MedNLI --user $username --ask-password https://physionet.org/files/mednli/1.0.0/mli_test_v1.jsonl
    mv data/MedNLI/physionet.org/files/mednli/1.0.0/mli_test_v1.jsonl data/MedNLI
    rm -r data/MedNLI/physionet.org
fi

if [ ! -f data/ProblemSummary/BioNLP2023-1A-Test.csv ]; then
    mkdir data/ProblemSummary
    wget -r -N -c -np -P data/ProblemSummary --user $username --ask-password https://physionet.org/files/bionlp-workshop-2023-task-1a/2.0.0/BioNLP2023-1A-Test.csv
    mv data/ProblemSummary/physionet.org/files/bionlp-workshop-2023-task-1a/2.0.0/BioNLP2023-1A-Test.csv data/ProblemSummary
    rm -r data/ProblemSummary/physionet.org
fi

if [ ! -f data/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx ]; then
    mkdir data/MeQSum
    wget -P data/MeQSum https://github.com/abachaa/MeQSum/blob/master/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx
fi

if [! -f data/LongHealth/benchmark_v5.json]; then
    mkdir data/LongHealth
    wget -P data/Longhealth https://github.com/kbressem/LongHealth/blob/main/data/benchmark_v5.json
fi