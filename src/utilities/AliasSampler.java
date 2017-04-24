package utilities;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

import org.json.JSONException;
import org.json.JSONObject;

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
	public static synchronized void writeObjectToFile(JSONObject js, String fileName, boolean append,boolean overwriteDuplicate) throws JSONException, IOException{
		boolean fileExists = Files.exists(Paths.get(fileName, new String[]{}));
		if(fileExists){
			fileExists = Utility.checkValidJsonFile(fileName);
		}
		if(append && fileExists){
			JSONObject oldObj = new JSONObject(Utility.readFile(fileName));
			for(String k:oldObj.keySet()){
				if(js.has(k)){
//					System.out.println("Key conflicting in overwriting file. Skipping it");
//					System.out.println("Conflict key: "+k);
//					js.remove(k);
					if(!overwriteDuplicate){
						js.put(k, oldObj.get(k));
					}
				}
				else{
					js.put(k,oldObj.get(k));
				}
			}
		}
		FileWriter fw = new FileWriter(fileName);
		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(js.toString(1));
		bw.close();		
	}
}
