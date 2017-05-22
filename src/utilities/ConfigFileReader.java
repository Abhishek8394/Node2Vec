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
}
