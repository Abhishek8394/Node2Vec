package main;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Vertex of a graph.
 * id is the name of the vertex, assumed integers. 
 * Children are the immediately connected nodes, an hashmap that keys children
 * based on their id and contains a pointer to them.
 * */
public class Node {
	private int id;
	ArrayList<Node> children;
	
	public Node(int id) {
		this.id = id;
		this.children = new ArrayList<Node>();
	}
	
	// Add a child node. Since it is a hash map all nodes with same id are assumed to be same
	public void addChild(Node child){
		this.children.add(child);
	}
	
	// Convenience function to check if given node is a child of this node
//	public boolean hasChild(Node child){
//		return children.containsKey(child.id);
//	}
//	public boolean hasChild(int childId){
//		return children.containsKey(childId);
//	}	
	// Returns child object with given id and returns null if not found.
//	public Node getChild(int id){
//		return children.getOrDefault(id, null);
//	}
	public Node getChild(int ind){
		return children.get(ind);
	}
	
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
//	public HashMap<Integer, Node> getChildren(){
//		return children;
//	}
	public Node[] getChildren(){
		return children.toArray(new Node[0]);
	}
	
	public int getChildCount(){
		return children.size();
	}
	
}
