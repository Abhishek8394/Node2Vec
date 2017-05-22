package main;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import org.json.JSONException;
import org.json.JSONObject;

import utilities.AliasSampler;
import utilities.Logger;
import utilities.Utility;

public class Node2vec {
	
	Graph graph;
	float p,q;
	HashMap<Integer,AliasSampler> nodeSamplers;
	HashMap<String,AliasSampler> edgeSamplers;
	
	// parameters for writing results and logs.
	String logDirectory;
	// decide if various info should be logged or not
	boolean enableLogging;
	// file to store node's alias samplers
	String nodeSampleLogFile;
	// file to store edge's alias samplers
	String edgeSampleLogFile;
	// file to store metadata info used during training
	String metaDataFile;
	// file to store some temporary data when handling
	// large amounts of nodes.
	// not actively used when under on-demand execution mode.
	String tmpDataFile;
	// How often to print some info. Not used in on demand execution mode.
	int summaryFrequency;
	// A counter to count number of times summary written. Used to 
	// split summaries across files.
	private int summaryWriteCounter;
	// File writing interface for adding meta data as generated
	private Logger metaDataLogger;
	// File writing interface for writing walks to file.
	private Logger walksFileWriter;
	// File name to store status of the walk generation procedure.
	// Used to resume walks across executions
	private String walkStatusFile;
	// Name of file to store the walks
	private String walksFile;
	// The JSON key to read when resuming execution.
	private static String LAST_PROCESSED_INDEX_KEY = "lastProcessedIndex";
	// All threads started by this class instance
	ArrayList<Thread> pendingTasks;
	// The size of number of edge samplers to keep in memory at a time.
	private int cleanUpThreshold = 30000;
	
	/**
	 * Constructor for  the Node2Vec.
	 * Parameters: Graph g ( the graph on which to run the algorithm )
	 * 			   float p ( the hyper parameter 'p' )
	 * 			   float q ( the hyper parameter 'q' )
	 * 			   boolean allow_log ( switch for allowing logging )
	 * 			   String log_dir ( directory for storing the walks )
	 * Return: Instance of Node2Vec class.
	 * */
	public Node2vec(Graph g, float p, float q, boolean allow_log, String log_dir){
		this.graph = g;
		this.p = p;
		this.q = q;
		this.enableLogging = allow_log;
		this.logDirectory = log_dir;
		this.nodeSampleLogFile = Utility.joinPaths(this.logDirectory,"nodeSamplers.txt");
		this.edgeSampleLogFile = Utility.joinPaths(this.logDirectory,"edgeSamplers");
		this.tmpDataFile = Utility.joinPaths(this.logDirectory,"tmp.txt");
		nodeSamplers = new HashMap<>();
		edgeSamplers = new HashMap<>();
		this.summaryFrequency = 5000;
		pendingTasks = new ArrayList<>();
		metaDataFile = Utility.joinPaths(this.logDirectory,"meta.txt");
		metaDataLogger = new Logger(metaDataFile);
		summaryWriteCounter = 0;
		walkStatusFile = Utility.joinPaths(this.logDirectory,"walk-status.txt");
		walksFile = Utility.joinPaths(this.logDirectory,"walks.txt");
		walksFileWriter = new Logger(walksFile);
	}
	
	/**
	 * Generates walks from a graph. Creates Alias Samplers on 
	 * a on-demand basis.
	 * Parameters: int num_walks ( number of walks to perform starting at each node)
	 * 			   int walk_length ( the length of each walk. Can be shorter if not feasible as 
	 * 								per the graph conditions )
	 * */
	public void generateWalks(int num_walks, int walk_length){
		Integer[] nodes = graph.getAllNodesAsArray();
		// Sort so that walks can be resumed by reading which node was 
		// processed last
		Arrays.sort(nodes);
		// See if can resume a previous walk procedure
		JSONObject walkStatus = getWalkStatus();
		int startIndex = 0;
		if(walkStatus!=null){
			// the node that should be started from 
			startIndex = (int) walkStatus.get(LAST_PROCESSED_INDEX_KEY);
		}
		else{
			// A new walk. Initialize parameters and write them.
			walkStatus = new JSONObject();
			walkStatus.put(LAST_PROCESSED_INDEX_KEY, startIndex);
			walkStatus.put("p",this.p);
			walkStatus.put("q",this.q);
			walkStatus.put("num_walks",num_walks);
			walkStatus.put("walk_length", walk_length);
		}
		// loop over all nodes starting from the node where the walk left off last time
		// or a new node in case of not resuming a walk
		for(int i = startIndex;i<nodes.length;i++){
			// Store walks in a buffer before writing them to the file
			ArrayList<String> walksBuffer = new ArrayList<>();
			// The node for which the walks are being generated
			Node curr = graph.getNode(nodes[i]);
			System.out.println("Node: "+curr.getId());
			// for each node we generate n number of walks.
			// The number of walks is a hyper parameter.
			for(int walkNum = 0; walkNum<num_walks;walkNum++){
				Node []walk = createSingleWalk(curr, walk_length);
				String walkStr = walkToString(walk);
				System.out.println(curr.getId()+": "+walkStr);
				walksBuffer.add(walkStr);
			}
			// For keeping a tab on the number of edge samplers being kept in memory.
			System.out.println(edgeSamplers.size()+" edge samplers");
			// update the walk status to resume in case execution interrupted.
			walkStatus.put(LAST_PROCESSED_INDEX_KEY, i);
			// Empty the walks buffer into the file.
			writeWalksToFile(walksBuffer,walkStatus);
			// If too many edge samplers generated, perform clean up
			if(edgeSamplers.size()>=cleanUpThreshold){
				System.out.println("Running maintenance...");
				// run the maintenance code for managing the edge samplers and other stuff.
				edgeSamplersMaintenance();
			}
		}
		
	}
	
	/**
	 * Run maintenance tasks to control the amount of edge samplers in memory.
	 * Also joins all the pending tasks created by the class.
	 * Basically a clean up utility to avoid crashes because of lack of resources
	 * Parameters: None
	 * Return: None
	 * */
	public void edgeSamplersMaintenance(){
		HashSet<String> batchEdges = new HashSet<>();
		// get list of the edges for which edge samplers are in memory
		batchEdges.addAll(edgeSamplers.keySet());
		// write edge samplers generated so far into memory.
		// Not doing currently because that too was time intensive
//		writeEdgeSamplesToFile(batchEdges);
		System.out.println("Writing everything to disk...");
		// clean up on pending tasks (threads)
		for(int i=0;i<pendingTasks.size();i++){
			try {
				pendingTasks.get(i).join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		// clear the queue of pending tasks
		pendingTasks.clear();
		// remove edges samplers from memory
		trimDownEdgeSamplers();
	}
	
	/**
	 * Reduce amount of edge samplers in memory. 
	 * TODO reove except all frequent ones
	 * Currently clears all edges samplers from memory
	 * Parameters: None
	 * Return: None
	 * */
	public void trimDownEdgeSamplers(){
		edgeSamplers.clear();
	}
	
	/**
	 * Write walks to a file.
	 * Parameters: ArrayList<String> walks ( all the walks in string form )
	 * 			   JSONObject walkStatusObject ( JSON object that is keeping track of the
	 * 											 status of walks )
	 * Return: None
	 * */
	public void writeWalksToFile(ArrayList<String> walks, JSONObject walkStatusObject){
		// create a reference to walks so that caller can release reference
		// and continue reusing the variable
		ArrayList<String> tmp = walks;
		// copy of walk status object
		JSONObject js = new JSONObject(walkStatusObject.toString());
		// using a thread to perform the writing task.
		Thread t = new Thread(new Runnable() {
			
			@Override
			public void run() {
				// for each walk write it into the file
				for(String s:tmp){
					try {
						walksFileWriter.log(s+"\n");
//						System.out.println(s);
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				// update the status file
				try {
					Utility.writeObjectToFile(walkStatusObject, walkStatusFile, true, true);
				} catch (JSONException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				System.out.println("Done writing batch of walks");
			}
		});
		// start the task in a separate thread and let the main thread 
		// continue with more walks
		t.start();
		// Store reference to thread for cleaning up later
		pendingTasks.add(t);
	}
	
	public Node[] createSingleWalk(Node n, int walk_length){
		Node[] walk = new Node[walk_length];
		walk[0] = n;
		for(int i=1;i<walk_length;i++){
			Node prev = walk[i-1];
			Node t = null;
			if(i==1){
				t = sampleNode(prev);
			}
			else{
				t = sampleEdge(walk[i-2],walk[i-1]);
			}
			walk[i]  = t;
			if(t==null){break;}
		}
		return walk;
	} 
	
	public Node sampleEdge(Node t, Node v){
		String edge = graph.encodeEdge(t.getId(), v.getId());
		if(edgeSamplers.containsKey(edge)){
			int index = edgeSamplers.get(edge).pickSample();
			return v.getChild(index);
		}
		float[] probs = transitionalProbs(t, v);
		float normalizingFactor = 0;
		for(int i=0;i<probs.length;i++){
			normalizingFactor+=probs[i];
		}
		for(int i=0;i<probs.length;i++){
			probs[i]/=normalizingFactor;
		}
		AliasSampler sampler = new AliasSampler(v.getChildCount());
		sampler.generateTables(probs);
		edgeSamplers.put(edge, sampler);
		int index = sampler.pickSample();
		// Sanity check.
//		if(!graph.hasEdge(v.getId(), v.getChild(index).getId())){
//			System.out.println("Error on edge "+graph.encodeEdge(v.getId(), v.getChild(index).getId()));
//			System.exit(-1);
//		}
		return v.getChild(index);
	}
	
	public Node sampleNode(Node n){
		if(n.getChildCount()==0){
			return null;
		}
		int id = n.getId();
		if(nodeSamplers.containsKey(id)){
			Integer index = nodeSamplers.get(id).pickSample();
			Node chosen = n.getChild(index);
			return chosen;
		}
		float[] probs = new float[n.getChildCount()];
		float normalizingFactor = 0;
		for(int i=0;i<probs.length;i++){
			Node nbr = n.getChild(i);
			probs[i] = graph.getEdgeWeight(id, nbr.getId());
			normalizingFactor+=probs[i];
		}
		for(int i=0;i<probs.length;i++){
			probs[i] = probs[i]/normalizingFactor;
		}
		Random r = new Random();
		float toss = r.nextFloat();
		float s = 0;
		for(int i=0;i<probs.length;i++){
			s+=probs[i];
			if(s>=toss){
				return n.getChild(i);
			}
		}
		return n.getChild(n.getChildCount()-1);
	}
	
	
	public void createNodeSamplers(){
		for(Integer id: graph.getAllNodes()){
			Node curr = graph.getNode(id);
			Node[] nbrs = curr.getChildren();
			float []probs = new float[nbrs.length];
			float normalizingFactor = 0;
			if(nbrs.length==0){
				System.out.println(curr.getId()+" has 0 nbrs");
			}
			for(int j= 0; j< nbrs.length;j++){
				Node cnbr = nbrs[j];
				probs[j] = graph.getEdgeWeight(curr.getId(), cnbr.getId());
				normalizingFactor+=probs[j];
			}
			for(int j=0;j<probs.length;j++){
				probs[j] /= normalizingFactor;
			}
			AliasSampler smplr = new AliasSampler(probs.length);
			smplr.generateTables(probs);
			nodeSamplers.put(curr.getId(), smplr);
		}		
	}
	
	// Processing all edges estimated to take 10 days. 
	// So process just a few at a time.
	public void createEdgeSamplers(){
		createEdgeSamplers(true, (int)graph.edges.size()/15);
	}
	
	public void createEdgeSamplers(boolean shouldBreak, int breakPoint){
		Set<String> edges = graph.getAllEdges();
		HashSet<String> batchEdges = new HashSet<>();
		System.out.println(edges.size()+" edges");
		int ctr=1;
		Date d = new Date();
		for(String s:edges){
			if(ctr%summaryFrequency == 0){
				System.out.println("writing results");
				writeEdgeSamplesToFile(batchEdges);
				batchEdges.clear();
			}
			System.out.println("Processing Edge: "+ ctr+" / "+edges.size());
			int[] nds = graph.decodeEdge(s);
			Node n1 = graph.getNode(nds[0]);
			Node n2 = graph.getNode(nds[1]);
			float[] probs = transitionalProbs(n1, n2);
			float normalizer = 0;
			for(int pIndex=0;pIndex<probs.length;pIndex++){
				normalizer+=probs[pIndex];
			}
			for(int pIndex=0;pIndex<probs.length;pIndex++){
				probs[pIndex]/=normalizer;
			}
			AliasSampler smplr = new AliasSampler(probs.length);
			smplr.generateTables(probs);
			if(edgeSamplers.containsKey(s)){
				System.out.println("Alert! "+s);
			}
			edgeSamplers.put(s, smplr);
			batchEdges.add(s);
			if(!graph.isDirected){
				probs = transitionalProbs(n2, n1);
				smplr = new AliasSampler(probs.length);
				String revEdge = graph.encodeEdge(n2.getId(), n1.getId());
				edgeSamplers.put(revEdge, smplr);
				batchEdges.add(revEdge);
			}
			ctr+=1;
			if(shouldBreak && ctr==breakPoint){
				break;
			}
		}
		Date d2 = new Date();
		System.out.println("took "+(d2.getTime()-d.getTime()));
		System.out.println("Cleaning up log buffers...");
		
		// make sure logging threads are done.
		for(int i=0;i<pendingTasks.size();i++){
			try {
				pendingTasks.get(i).join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		System.out.println("All log threads done.");
	}
	
	/* Calculate two level markov chain as per formula in paper. 
	 * t=prev node, v=curr node, x = neighbors of n2
	 */
	public float[] transitionalProbs(Node t, Node v){
		Node[] vNbrs = v.getChildren();
		float[] probs = new float[vNbrs.length];
		for(int j=0;j<vNbrs.length;j++){
			Node currNbr = vNbrs[j];
			float w = graph.getEdgeWeight(v.getId(), currNbr.getId());
			if(currNbr.getId() == t.getId()){
				// Since bias = 1/p for d(t,x)=0 
				probs[j] = w / p;
			}
			else if(graph.hasEdge(t.getId(), currNbr.getId())){
				// Since bias = 1  for d(t,x)==1
				probs[j] = w;
			}
			else{
				// Since bias = 1/q otherwise
				probs[j] = w/q;
			}
		}
		return probs;
	}
	
	public void writeEdgeSamplesToFile(HashSet<String> batchEdges){
		String[] tmpKeys = batchEdges.toArray(new String[]{});
		Thread t = new Thread(new Runnable() {
			
			@Override
			public void run() {
				HashMap<String, AliasSampler> samplersToLog = new HashMap<String,AliasSampler>();
				for(String s:tmpKeys){
					samplersToLog.put(s, edgeSamplers.get(s));
				}
				if(tmpKeys.length==0){
					return;
				}
				JSONObject js = new JSONObject(samplersToLog);
				try {
					String filename = generateEdgeSampleFileName(true);
					Utility.writeObjectToFile(js, filename, true);
					Utility.log("Node2Vec", "wrote edge sample to file");
					System.out.println("Wrote edgelog samplers: "+samplersToLog.keySet().size());

					JSONObject metaObj = new JSONObject();
					metaObj.append("edges", samplersToLog.keySet());
					metaObj.append("file", filename);
//					System.out.println(metaObj.toString(1));
					metaDataLogger.log(metaObj.toString(1)+",\n");
				} catch (JSONException | IOException e) {
					// TODO Auto-generated catch block
					System.out.println("Error writing to edge sample file");
					e.printStackTrace();
				}
			}
		});
		t.start();
		pendingTasks.add(t);
	}
	
	public String generateEdgeSampleFileName(boolean newFile){
		if(newFile){
			summaryWriteCounter+=1;
		}
		return getEdgeSampleLogFile() + "-"+summaryWriteCounter+".txt";
	}
	
	public String walkToString(Node []walk){
		StringBuffer sb = new StringBuffer();
		for(int i=0;i<walk.length;i++){
			if(walk[i]==null){break;}
			sb.append(walk[i].getId()+":");
		}
		return sb.toString();
	}
	
	public JSONObject getWalkStatus(){
		JSONObject walkStatus = null;
		try {
			walkStatus = new JSONObject(Utility.readFile(walkStatusFile));
		}
		catch(FileNotFoundException e){
			return walkStatus;
		}
		catch (JSONException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return walkStatus;
	}
	
	public Graph getGraph() {
		return graph;
	}

	public void setGraph(Graph graph) {
		this.graph = graph;
	}

	public float getP() {
		return p;
	}

	public void setP(float p) {
		this.p = p;
	}

	public float getQ() {
		return q;
	}

	public void setQ(float q) {
		this.q = q;
	}

	public HashMap<Integer, AliasSampler> getNodeSamplers() {
		return nodeSamplers;
	}

	public void setNodeSamplers(HashMap<Integer, AliasSampler> nodeSamplers) {
		this.nodeSamplers = nodeSamplers;
	}

	public HashMap<String, AliasSampler> getEdgeSamplers() {
		return edgeSamplers;
	}

	public void setEdgeSamplers(HashMap<String, AliasSampler> edgeSamplers) {
		this.edgeSamplers = edgeSamplers;
	}

	public String getLogDirectory() {
		return logDirectory;
	}

	public void setLogDirectory(String logDirectory) {
		this.logDirectory = logDirectory;
	}

	public boolean isEnableLogging() {
		return enableLogging;
	}
	
	public void setEnableLogging(boolean enableLogging) {
		this.enableLogging = enableLogging;
	}

	public String getNodeSampleLogFile() {
		return nodeSampleLogFile;
	}

	public void setNodeSampleLogFile(String nodeSampleLogFile) {
		this.nodeSampleLogFile = nodeSampleLogFile;
	}

	public int getSummaryFrequency() {
		return summaryFrequency;
	}

	public void setSummaryFrequency(int summaryFrequency) {
		this.summaryFrequency = summaryFrequency;
	}

	public String getEdgeSampleLogFile() {
		return edgeSampleLogFile;
	}

	public void setEdgeSampleLogFile(String edgeSampleLogFile) {
		this.edgeSampleLogFile = edgeSampleLogFile;
	}

	public String getTmpDataFile() {
		return tmpDataFile;
	}

	public void setTmpDataFile(String tmpDataFile) {
		this.tmpDataFile = tmpDataFile;
	}

	public String getMetaDataFile() {
		return metaDataFile;
	}

	public void setMetaDataFile(String metaDataFile) {
		this.metaDataFile = metaDataFile;
	}

	public ArrayList<Thread> getPendingTasks() {
		return pendingTasks;
	}

	public void setPendingTasks(ArrayList<Thread> pendingTasks) {
		this.pendingTasks = pendingTasks;
	}

	public String getWalkStatusFile() {
		return walkStatusFile;
	}

	public void setWalkStatusFile(String walkStatusFile) {
		this.walkStatusFile = walkStatusFile;
	}

	public String getWalksFile() {
		return walksFile;
	}

	public void setWalksFile(String walksFile) {
		this.walksFile = walksFile;
	}
}
