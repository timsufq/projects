package test;
public class MyThread extends Thread
{
	public MyThread(String threadName)
	{
		super(threadName);
	}
	public void run()
	{
		for (int i = 1; i < 5; i++)
		{
			try
			{
				Thread.sleep(500);
				System.out.println(this.getName() + ":" + i);
			}
			catch(InterruptedException e)
			{
				System.out.println(e);
			}
		}
	}
	public static void main(String args[])
	{
		new MyThread("Thread 1").start();
		new MyThread("Thread 2").start();
	}
}
