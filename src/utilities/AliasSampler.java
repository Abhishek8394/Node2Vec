package utilities;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

public class AliasSampler {
	public ArrayList<Float> probs;		// probability table
	public int n;						// number of elements
	public ArrayList<Integer> kTable;	// k-table
	
	public AliasSampler(int n){
		this.n = n;
		probs = new ArrayList<>();
		kTable = new ArrayList<>();
	}
	
	public void generateTables(float[] probabilities){
		Queue<Integer> smalls = new LinkedList<Integer>();
		Queue<Integer> large = new LinkedList<Integer>();
		for(int i=0;i<probabilities.length;i++){
			float p = probabilities.length * probabilities[i];
			probs.add(p);
			kTable.add(0);
			if(p<1){
				smalls.add(i);
			}
			else{
				large.add(i);
			}
		}
		while(smalls.size()>0 && large.size()>0 ){
			int s = smalls.remove();
			int l = large.remove();
			kTable.set(s, l);
			probs.set(l, probs.get(l)+probs.get(s)-1);
			if(probs.get(l)<1){
				smalls.add(l);
			}
			else{
				large.add(l);
			}
		}
	}
	
	public Integer pickSample(){
		int index = 0;
		Random random = new Random();
		float p = random.nextFloat();
		int i = (int)Math.floor(probs.size() * p);
		float y = probs.size()*p - i;
		if(y<probs.get(i)){
			index = i;
		}
		else{
			index = kTable.get(i);
		}
		return index;
	}

	public ArrayList<Float> getProbs() {
		return probs;
	}

	public void setProbs(ArrayList<Float> probs) {
		this.probs = probs;
	}

	public int getN() {
		return n;
	}

	public void setN(int n) {
		this.n = n;
	}

	public ArrayList<Integer> getKTable() {
		return kTable;
	}

	public void setkTable(ArrayList<Integer> kTable) {
		this.kTable = kTable;
	} 
	
	public void generateTablesShort(float[] probabilities){
		Queue<Integer> sm = new LinkedList<Integer>();
		Queue<Integer> lg = new LinkedList<Integer>();
		for(int i=0;i<probabilities.length;i++){
			float p = probabilities.length * probabilities[i];
			probs.add(p);
			kTable.add(0);
			if(p<1){
				sm.add(i);
			}
			else{
				lg.add(i);
			}
		}
		while(sm.size()>0 && lg.size()>0 ){
			int s = sm.remove();
			int l = lg.remove();
			kTable.set(s, l);
			probs.set(l, probs.get(l)+probs.get(s)-1);
			if(probs.get(l)<1){
				sm.add(l);
			}
			else{
				lg.add(l);
			}
		}
	}
}
