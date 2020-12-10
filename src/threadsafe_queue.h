#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <exception>
#include<stack>
class threadsafe_queue
{
  private:
     mutable std::mutex mut;

     std::condition_variable data_cond;
  public:
     std::queue<cv::Mat> data_queue;
  public:
     threadsafe_queue(){}
     threadsafe_queue(threadsafe_queue const& other)
     {
         std::lock_guard<std::mutex> lk(other.mut);
         data_queue=other.data_queue;
     }
     void push(cv::Mat &new_value)//入队操作
     {
         std::lock_guard<std::mutex> lk(mut);
         if(data_queue.size() >= 300 )
         {
        	//  std::cout<< "lost mat" <<std::endl;
        	 data_cond.notify_one();
        	 return;
         }

         data_queue.push(new_value);
         data_cond.notify_one();
     }
     void wait_and_pop(cv::Mat& value)//直到有元素可以删除为止
     {
         std::unique_lock<std::mutex> lk(mut);
         data_cond.wait(lk,[this]{return !data_queue.empty();});
         value=data_queue.front();
         data_queue.pop();
     }
     std::shared_ptr<cv::Mat> wait_and_pop()
     {
         std::unique_lock<std::mutex> lk(mut);
         data_cond.wait(lk,[this]{return !data_queue.empty();});
         std::shared_ptr<cv::Mat> res(std::make_shared<cv::Mat>(data_queue.front()));
         data_queue.pop();
         return res;
     }
     bool try_pop(cv::Mat& value)//不管有没有队首元素直接返回
     {
         std::lock_guard<std::mutex> lk(mut);
         if(data_queue.empty())
             return false;
         value=data_queue.front();
         data_queue.pop();
         return true;
     }
     std::shared_ptr<cv::Mat> try_pop()
     {
         std::lock_guard<std::mutex> lk(mut);
         if(data_queue.empty())
             return std::shared_ptr<cv::Mat>();
         std::shared_ptr<cv::Mat> res(std::make_shared<cv::Mat>(data_queue.front()));
         data_queue.pop();
         return res;
     }
     bool empty() const
     {
         std::lock_guard<std::mutex> lk(mut);
         return data_queue.empty();
     }
     int size(){
            return data_queue.size();
        }
};



template<typename T>
class threadsafe_stack
{
	private:
		std::stack<T> data;
		mutable std::mutex m;
	public:
		threadsafe_stack(){}
		threadsafe_stack(const threadsafe_stack& other)
		{
			std::lock_guard<std::mutex> lock(other.m);
			data=other.data;
		}
		threadsafe_stack& operator=(const threadsafe_stack&) = delete;
		void push(T new_value)
		{
			std::lock_guard<std::mutex> lock(m);
            if(data.size() >= 300 )
         {
            data.pop(); 
         }
			data.push(std::move(new_value)); 
		}
		std::shared_ptr<T> pop()//top和pop合并，采用shared_ptr返回栈顶元素防止元素构造时发生异常
		{
			std::lock_guard<std::mutex> lock(m);
				std::shared_ptr<T> const res(std::make_shared<T>(std::move(data.top())));//make_shared比shared_ptr直接构造效率高 
			data.pop(); 
			return res;
		}
		void pop(T& value)//采用参数引用返回栈顶元素
		{
			std::lock_guard<std::mutex> lock(m);
			value=std::move(data.top()); 
			data.pop(); 
		}
		bool empty() const
		{
			std::lock_guard<std::mutex> lock(m);
			return data.empty();
		}
        int size(){
            return data.size();
        }
};
