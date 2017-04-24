package utilities;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;

import org.json.JSONException;
import org.json.JSONObject;

import main.Graph;
import main.Node;

public class DatasetLoaders {

	public static void loadNodes(String nodefile, Graph graph) throws IOException{
//		HashMap<Integer, Node> nodelist = new HashMap<Integer, Node>();
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(nodefile)));
		String line="";
		while((line=reader.readLine())!=null){
			int id = Integer.parseInt(line);
			Node n = new Node(id);
			graph.addNode(n);
//			nodelist.put(id, new Node(id));
		}
//		return nodelist;
	}
	
	
	public static void loadEdges(String edgesFile, Graph graph) throws IOException{
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(edgesFile)));
		String line = "";
		while((line=reader.readLine())!=null){
			String[] edge = line.split(",");
			int v1 = Integer.parseInt(edge[0]);
			int v2 = Integer.parseInt(edge[1]);
			
			graph.addEdge(v1, v2);
			graph.getNode(v1).addChild(graph.getNode(v2));
			if(!graph.isDirected){
				graph.getNode(v2).addChild(graph.getNode(v1));
			}
		}
		reader.close();
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
