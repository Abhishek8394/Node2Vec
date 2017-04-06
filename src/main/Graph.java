package main;

import java.util.HashMap;
import java.util.Set;

public class Graph {
	public HashMap<Integer, Node> nodes;
	public HashMap<String, Float> edges;
	public boolean isDirected;
	public String encodingDelim = ",";
	
	public Graph(boolean is_directed){
		isDirected = is_directed;
		nodes = new HashMap<>();
		edges = new HashMap<>();
	}
	
	public void addEdge(int v1, int v2, float weight){
		edges.put(encodeEdge(v1, v2), weight);
	}
	public void addEdge(int v1, int v2){
		edges.put(encodeEdge(v1, v2), (float) 1);
	}
	
	public void addNode(Node n){
		nodes.put(n.getId(),n);
	}
	
	public float getEdgeWeight(int v1, int v2){
		String k = encodeEdge(v1, v2);
		if(edges.containsKey(k)){
			return edges.get(k);
		}
		k = encodeEdge(v2, v1);
		if(!isDirected && edges.containsKey(k)){
			return edges.get(k);
		}
		return 0;
	}
	
	public boolean hasEdge(int v1, int v2){
		String k = encodeEdge(v1, v2);
		if(edges.containsKey(k)){
			return true;
		}
		if(!isDirected && edges.containsKey(encodeEdge(v2, v1))){
			return true;
		}
		return false;
	}
	
	public Node getNode(int id){
		return nodes.getOrDefault(id, null);
	}
	
	public String encodeEdge(int v1, int v2){
		return String.format("%d%s%d", v1, encodingDelim, v2);
	}
	
	public int[] decodeEdge(String e){
		String[] nds = e.split(encodingDelim);
		int[] nodeIds = new int[nds.length];
		for(int i=0;i<nds.length;i++){
			nodeIds[i] = Integer.parseInt(nds[i]);
		}
		return nodeIds;
	}
	
	public Set<Integer> getAllNodes(){
		return nodes.keySet();
	}
	
	public Integer[] getAllNodesAsArray(){
		return nodes.keySet().toArray(new Integer[0]);
	}
	
	public Set<String> getAllEdges(){
		return edges.keySet();
	}
	
}
