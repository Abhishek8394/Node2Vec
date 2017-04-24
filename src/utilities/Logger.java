package utilities;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.json.JSONException;
import org.json.JSONObject;

public class Logger {
	public String logFile;
	private BufferedWriter writer;
	
	public Logger(String logfile){
		logFile = logfile;
		FileWriter fw;
		try {
			fw = new FileWriter(logFile, true);
			writer = new BufferedWriter(fw);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void log(String msg) throws IOException{
		writer.write(msg);
		writer.flush();
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
	public static synchronized void writeObjectsToFile(JSONObject js, String fileName, boolean append,boolean overwriteDuplicate) throws JSONException, IOException{
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
