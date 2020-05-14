package deadLock;
public class DeadLock
{
	static final Object lock1=new Object();
	static final Object lock2=new Object();
	public static void main(String arg0[])
	{
		System.out.println("Starting: "+Thread.currentThread());
		new Thread
		(
			new Runnable()
			{
				public void run()
				{
					System.out.println("I am run()! ");
					synchronized(lock1)
					{
						try
						{
							Thread.sleep(500);
						}catch(InterruptedException e)
						{
							e.printStackTrace();
						}
						
						synchronized(lock2)
						{
							System.out.println("Running: "+Thread.currentThread());
						}
					}
				}
			}
		).start();
		synchronized(lock2)
		{
			System.out.println("I am main()! ");
			try
			{
				Thread.sleep(500);
			}catch(InterruptedException e)
			{
				e.printStackTrace();
			}
			synchronized(lock1)
			{
				System.out.println("Finishing: "+Thread.currentThread());
			}
		}
	}
}
