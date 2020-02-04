import com.github.cage.Cage;
import com.github.cage.GCage;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.IntStream;

public class Generator {
    private static final int N_THREADS = 16;
    private static final ExecutorService pool = Executors.newFixedThreadPool(N_THREADS);

    public static void main(String[] args) {
        Cage cage = new GCage();
        IntStream.range(0, 90000).forEach(i -> {
            Runnable gen = () -> {
                try {
                    generate(cage);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            };
            pool.execute(gen);
        });
        pool.shutdown();
    }

    private static void generate(Cage cage) throws IOException {
        var number = new Random().nextInt(100);
        var token = cage.getTokenGenerator().next();
        OutputStream os = new FileOutputStream(String.format("images/%02d_%s.png", number, token), false);
        cage.draw(String.format("%02d", number), os);
        os.close();
    }
}
