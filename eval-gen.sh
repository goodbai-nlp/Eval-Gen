#:<<!
gold=$1
pred=$2
#!
echo "Evaluating BLEU score ..."
python amr_eval_gen.py --in-tokens $pred --in-reference-tokens $gold

java -jar meteor-1.5/meteor-1.5.jar $pred $gold > $pred.meteor
tail -n 10 $pred.meteor

