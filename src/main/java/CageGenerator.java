import com.github.cage.*;

import java.io.*;
import java.util.ArrayList;

public class CageGenerator {

    public static void main(String[] args) throws IOException{


        //Clear out k,v store before new batch image generation
        File keyStore = new File("keys.txt");
        if(keyStore.exists()){
            keyStore.delete();
            keyStore.createNewFile();
        }

        //@Deprecated, but still usable
        keyStore.setWritable(true);
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(keyStore));


        for (int i = 0; i < 1000; ++i) {
            generate(new Cage(), i, "test", ".jpg", null, bufferedWriter);
        }


        bufferedWriter.close();
    }


    public static void generate(Cage cage,
                                int num,
                                String namePrefix,
                                String namePostfix,
                                String text,
                                BufferedWriter bufferedWriter)
            throws IOException {

        //Initialize file output paths

        String path = new File(".").getAbsolutePath() + "/test/" + namePrefix + num +  namePostfix;
        OutputStream fos = new FileOutputStream(path, false);


        //Generate random captcha tokens and store file name and
        //tokens into "keys.txt"
        //String token;
        try {
            String token = cage.getTokenGenerator().next();
            token = token.substring(0, 5); // length five captcha
            cage.draw(token, fos);
            bufferedWriter.write(namePrefix + num + namePostfix + "," + token + "\n");

        }
        finally { fos.close(); }
    }

}
