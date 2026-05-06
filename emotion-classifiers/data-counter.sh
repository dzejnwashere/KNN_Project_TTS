#!/bin/bash

echo "bea_Amused"
find bea_Amused/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls bea_Amused | wc -l

echo "bea_Angry"
find bea_Angry/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls bea_Angry | wc -l

echo "bea_Disgusted"
find bea_Disgusted/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls bea_Disgusted | wc -l

echo "bea_Neutral"
find bea_Neutral/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls bea_Neutral | wc -l

echo "jenie_Amused"
find jenie_Amused/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls jenie_Amused | wc -l

echo "jenie_Angry"
find jenie_Angry/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls jenie_Angry | wc -l

echo "jenie_Disgusted"
find jenie_Disgusted/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls jenie_Disgusted | wc -l

echo "jenie_Neutral"
find jenie_Neutral/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls jenie_Neutral | wc -l

echo "josh_Amused"
find josh_Amused/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls josh_Amused | wc -l

echo "josh_Neutral"
find josh_Neutral/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls josh_Neutral | wc -l

echo "sam_Amused"
find sam_Amused/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls sam_Amused | wc -l

echo "sam_Angry"
find sam_Angry/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls sam_Angry | wc -l

echo "sam_Disgusted"
find sam_Disgusted/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls sam_Disgusted | wc -l

echo "sam_Neutral"
find sam_Neutral/ -name "*.wav" -exec soxi -D {} \; | awk '{sum += $1} END {print sum/60 " minutes"}'
ls sam_Neutral | wc -l
