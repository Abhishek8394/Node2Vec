package utilities;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;

import org.json.JSONException;
import org.json.JSONObject;

public class ConfigFileReader {
	
	HashMap<String, Object> configObject;
	
	public ConfigFileReader(String filename) throws IOException{
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
		String fileContents = Utility.readFile(reader);
		JSONObject configJson = new JSONObject(fileContents);
		configObject = new HashMap<>();
		Iterator<String> keys = configJson.keys();
		while(keys.hasNext()){
			String k = keys.next();
			Object val = configJson.get(k);
			configObject.put(k,val);
		}
		reader.close();
	}
	
	public Object getConfigOption(String key){
		return configObject.getOrDefault(key,null);
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
