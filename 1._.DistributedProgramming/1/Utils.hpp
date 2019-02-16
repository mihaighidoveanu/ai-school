//
//  Utils.hpp
//  Lab
//
//  Created by Ciprian on 10/13/18.
//  Copyright © 2018 Ciprian. All rights reserved.
//

#ifndef Utils_hpp
#define Utils_hpp

#include <chrono>
#include <thread>
#include <iostream>
#include <sstream>
#include <stdlib.h>

class Utils
{
public:
    static void SleepMs(const int timeInMilliseconds)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(timeInMilliseconds));
    }


    static void SleepRandomTime(const int min, const int max)
    {
        const int millisecondsToSleep = rand() % (max - min + 1) + min;
        SleepMs(millisecondsToSleep);
    }


    // Why do we need this function ?
    // If we do std::cout from different threads, their results will be intercalated
    // So we must synchronize the output
    static void writeConsoleMessageSynchronized(const std::ostringstream& textToWrite)
    {
        static std::mutex s_consoleMutex; // Locally, shared by all threads
        
        std::lock_guard<std::mutex> guard(s_consoleMutex); // Modern way of locking, deactivates the lock automatically on destructor in case that you forget
        std::cout << textToWrite.str();
    }
};

#endif /* Utils_hpp */
