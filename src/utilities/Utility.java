package utilities;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * Class containing basic utilities
 * */
public class Utility {

	public static Logger logger = null;;
	public static String LOG_FILE = "logs.txt";
	
	public static String readFile(String filename) throws IOException{
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
		String data = readFile(reader);
		reader.close();
		return data;
	}

	// Read the contents of a file and return it as a string.
	// Does not close the reader, leaves it to the caller
	public static String readFile(BufferedReader reader) throws IOException{
		StringBuffer sb=new StringBuffer();
		String line="";
		while((line=reader.readLine())!=null){
			sb.append(line);
			
		}
		return sb.toString();
	}
	
	public static String joinPaths(String ...paths){
		String p = String.join(File.separator, paths);
		return p;
	}
	
	public static void makeDirectory(String path){
		Path p = Paths.get(path, new String[]{});
		if(!Files.exists(p)){
			File f = new File(path);
			boolean success = f.mkdirs();
			if(!success){
				System.out.println("failed to make directory: "+path);
			}
		}
	}
	
	public static void writeObjectToFile(JSONObject js, String fileName, boolean append) throws JSONException, IOException{
		writeObjectToFile(js, fileName, append, false);
	}
	
	public static synchronized void writeObjectToFile(JSONObject js, String fileName, boolean append,boolean overwriteDuplicate) throws JSONException, IOException{
		boolean fileExists = Files.exists(Paths.get(fileName, new String[]{}));
		if(fileExists){
			fileExists = checkValidJsonFile(fileName);
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
	
	public static synchronized void log(String tag, String msg){
		if(logger==null){
			logger = new Logger(LOG_FILE);
		}
		try {
			Date date = new Date();
			DateFormat df = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
			StringBuffer logStr = new StringBuffer();
			logStr.append("["+df.format(date)+"] ");
			logStr.append(tag+": ");
			logStr.append(msg);
			logger.log(logStr.toString()+"\n");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static boolean checkValidJsonFile(String filename){
		try{
			JSONObject js = new JSONObject(readFile(filename));
			return true;
		}
		catch(Exception e){
			return false;
		}
	}

	public static String getLOG_FILE() {
		return LOG_FILE;
	}

	public static void setLOG_FILE(String lOG_FILE) {
		LOG_FILE = lOG_FILE;
		logger = new Logger(LOG_FILE);
	}
	
	public static void copyFile(String src, String dest) throws IOException{
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(src)));
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(dest)));
		String line;
		while((line=reader.readLine())!=null){
			writer.write(line+"\n");
		}
		writer.close();
		reader.close();
	}
}
