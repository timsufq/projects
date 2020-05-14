package threadTest;

public class TestThread extends Thread
{
	private int identity;
	public TestThread(int id)
	{
		this.identity=id;
	}
	public void run()
	{
		for(int i=1; i<6; i++)
		{
			System.out.println(identity+" Thread: "+i+" ActiveCount: "+Thread.activeCount());
		}
	}
	public static void main(String arg0[])
	{
		TestThread t1=new TestThread(1);
		TestThread t2=new TestThread(2);
		synchronized(Thread.currentThread()){t1.start();t2.start();}
		//t1.start();//t1.run();
		//t2.start();//t2.run();
	}
}
