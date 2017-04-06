package utilities;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

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
	
}
