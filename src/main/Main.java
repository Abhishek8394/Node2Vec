package main;

import java.io.IOException;
import java.util.Date;

import org.json.JSONObject;

import utilities.ConfigFileReader;
import utilities.DatasetLoaders;
import utilities.Utility;

public class Main {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		String configFile = args[0];
		ConfigFileReader cf = new ConfigFileReader(configFile);
		String nodesFile = (String) cf.getConfigOption("nodes_file");
		String node_edges_file = (String) cf.getConfigOption("nodes_edges_file");
		boolean isDirected = (boolean)cf.getConfigOption("is_directed");
		float p = (float)((double)cf.getConfigOption("p"));
		float q = (float)((double)cf.getConfigOption("q"));
		String logdir = (String)cf.getConfigOption("log_dir");
		String logfileName = "logs.txt";
		boolean allow_log = (boolean)cf.getConfigOption("allow_logging");
		int num_walks = (int)cf.getConfigOption("num_walks");
		int walk_length = (int)cf.getConfigOption("walk_length");
		Date d = new Date();
		String time = d.getTime()+"";
		String sessionLogDir = Utility.joinPaths(logdir,time);
		Utility.makeDirectory(sessionLogDir);
		Utility.copyFile(configFile, Utility.joinPaths(sessionLogDir,"config.txt"));
		logfileName = Utility.joinPaths(sessionLogDir,logfileName); 
		Utility.setLOG_FILE(logfileName);
		

		Graph graph = new Graph(isDirected);
		System.out.println("Loading nodes");
		DatasetLoaders.loadNodes(nodesFile,graph);
		System.out.println("Loading edges");
		DatasetLoaders.loadEdges(node_edges_file, graph);
		Node2vec n2v = new Node2vec(graph,p,q,allow_log, sessionLogDir);		
		System.out.println("Creating node samplers");
//		n2v.createNodeSamplers();
//		JSONObject j = new JSONObject(n2v.getNodeSamplers());
//		Utility.writeObjectToFile(j, n2v.getNodeSampleLogFile(), true);
		System.out.println("Creating edge samplers");
		
//		n2v.createEdgeSamplers();
		n2v.generateWalks(num_walks,walk_length);
//		for(Integer i:graph.getAllNodes()){
//			if(graph.hasEdge(1, i))
//			{
//				System.out.println(i);
//			}
//		}
//		float[] p = new float[10];
//		Random r = new Random();
//		float s = 0;
//		for(int i=0;i<p.length;i++){
//			p[i] = 1;
//			s+=p[i];
//		}
//		for(int i=0;i<p.length;i++){
//			p[i] = p[i]/s;
////			System.out.print(String.format("%.2f", p[i])+" ");
//		}
//		AliasSampler as = new AliasSampler(p.length);
//		as.generateTables(p);
//		for(int i=0;i<5;i++){
//			System.out.println(as.pickSample());
//		}

	}

}
